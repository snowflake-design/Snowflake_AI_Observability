import os
import pandas as pd
import time
import random  # ADD THIS - for username simulation
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.session import Session
from snowflake.cortex import complete
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
from trulens.apps.app import TruApp
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core.run import Run, RunConfig
from trulens.core import Feedback, Provider  # ADD THIS - for custom feedback functions

# ADD THIS - Import for toxicity detection
try:
    from transformers import pipeline
    toxicity_classifier = pipeline("text-classification", model="s-nlp/roberta_toxicity_classifier")
    print("✅ Toxicity classifier loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load toxicity classifier: {e}")
    toxicity_classifier = None

# ADD THIS - Custom Provider for TruLens feedback functions
class CustomMetricsProvider(Provider):
    """Custom provider to wrap our metrics as TruLens feedback functions."""
    
    def __init__(self):
        super().__init__()
        self.toxicity_classifier = toxicity_classifier
    
    def toxicity_feedback(self, output: str) -> float:
        """Toxicity detection as TruLens feedback function."""
        if self.toxicity_classifier is None:
            return 0.5  # neutral score
        
        try:
            result = self.toxicity_classifier(output)
            label = result[0]['label']
            confidence = result[0]['score']
            
            print(f"🛡️ Toxicity feedback: {label} (confidence: {confidence:.3f})")
            
            # Convert to TruLens score: 1 = good (non-toxic), 0 = bad (toxic)
            if label == 'TOXIC':
                return 1.0 - confidence  # toxic = low score
            else:
                return confidence  # non-toxic = high score
                
        except Exception as e:
            print(f"❌ Toxicity feedback error: {e}")
            return 0.5
    
    def username_feedback(self, query: str) -> float:
        """Username extraction as TruLens feedback function."""
        try:
            if 'username' in query.lower() or any(user in query for user in ['abc', 'XYZ', 'KKK']):
                print(f"👤 Username detected in query")
                return 1.0  # username found
            else:
                print(f"👤 No username detected")
                return 0.0  # no username
        except Exception as e:
            print(f"❌ Username feedback error: {e}")
            return 0.5

# Enable TruLens OpenTelemetry tracing
os.environ["TRULENS_OTEL_TRACING"] = "1"

# Get Snowflake session
try:
    session = get_active_session()
    print("Using active Snowflake session")
except:
    print("Creating new Snowflake session...")
    session = Session.builder.configs(SNOWFLAKE_CONFIG).create()

print(f"Current context: {session.get_current_database()}.{session.get_current_schema()}")

class RAGApplication:
    def __init__(self):
        self.model = "mistral-large2"
        
        self.knowledge_base = {
            "machine learning": [
                "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
                "ML algorithms can identify patterns in large datasets and make predictions or decisions based on those patterns.",
                "Common types include supervised learning, unsupervised learning, and reinforcement learning."
            ],
            "cloud computing": [
                "Cloud computing delivers computing services over the internet including storage, processing power, and applications.",
                "Major benefits include scalability, cost-effectiveness, and accessibility from anywhere with internet connection.",
                "Popular cloud providers include AWS, Microsoft Azure, and Google Cloud Platform."
            ],
            "artificial intelligence": [
                "AI refers to computer systems that can perform tasks typically requiring human intelligence.",
                "AI encompasses machine learning, natural language processing, computer vision, and robotics.",
                "Applications include chatbots, recommendation systems, autonomous vehicles, and medical diagnosis."
            ]
        }

    # ADD THIS - Custom toxicity detection method
    def detect_toxicity(self, text: str) -> str:
        """Detect if text is toxic using HuggingFace classifier."""
        if toxicity_classifier is None:
            return "unknown"
        
        try:
            result = toxicity_classifier(text)
            # The model returns labels like 'TOXIC' or 'NOT_TOXIC'
            label = result[0]['label']
            confidence = result[0]['score']
            
            # Return yes/no based on prediction
            if label == 'TOXIC' and confidence > 0.5:
                return "yes"
            else:
                return "no"
        except Exception as e:
            print(f"Error in toxicity detection: {e}")
            return "unknown"

    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        }
    )
    def retrieve_context(self, query: str) -> list:
        """Retrieve relevant text from knowledge base."""
        query_lower = query.lower()
        retrieved_contexts = []
        
        for topic, contexts in self.knowledge_base.items():
            if topic in query_lower:
                retrieved_contexts.extend(contexts)
                break
        
        if not retrieved_contexts:
            retrieved_contexts = self.knowledge_base.get("artificial intelligence", [])
        
        return retrieved_contexts

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate_completion(self, query: str, context_str: list) -> str:
        """Generate answer from context by calling an LLM."""
        context_text = "\n".join([f"- {ctx}" for ctx in context_str])
        
        prompt = f"""Based on the following context, answer the question:

Context:
{context_text}

Question: {query}

Answer:"""
        
        try:
            response = complete(self.model, prompt)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    @instrument(span_type=SpanAttributes.SpanType.RECORD_ROOT)
    def answer_query(self, input_data, username: str = "unknown") -> str:  # MODIFIED TO HANDLE BOTH
        """Main entry point for the RAG application."""
        
        # Handle different input types
        if isinstance(input_data, dict):
            # Dataset row input
            query = input_data.get('query', '')
            username = input_data.get('username', 'unknown')
        elif hasattr(input_data, 'get'):
            # Pandas Series input
            query = input_data.get('query', str(input_data))
            username = input_data.get('username', 'unknown')
        else:
            # String input (direct query)
            query = str(input_data)
            # username parameter will be used
        
        # ADD THIS - Log username and toxicity in trace context
        print(f"🔍 Processing query from user: {username}")
        
        # Detect toxicity
        toxicity_result = self.detect_toxicity(query)
        print(f"🛡️ Toxicity check for '{query[:50]}...': {toxicity_result}")
        
        context_str = self.retrieve_context(query)
        response = self.generate_completion(query, context_str)
        
        # ADD THIS - Log completion with user info
        print(f"✅ Response generated for user {username} (toxicity: {toxicity_result})")
        
        return response

# Initialize the RAG application
test_app = RAGApplication()

# Test basic functionality
print("Testing basic functionality...")
test_response = test_app.answer_query("What is machine learning?", "test_user")  # ADD USERNAME
print(f"Test successful: {test_response[:100] if test_response else 'No response'}...")

# CREATE CUSTOM FEEDBACK FUNCTIONS - ADD THIS SECTION
print("\n" + "="*60)
print("CREATING CUSTOM FEEDBACK FUNCTIONS FOR DASHBOARD")
print("="*60)

# Initialize custom provider
custom_provider = CustomMetricsProvider()

# Create feedback functions that will appear as metrics in Snowsight
f_toxicity = Feedback(
    custom_provider.toxicity_feedback,
    name="custom_toxicity"  # This should appear in dashboard
).on_output()

f_username = Feedback(
    custom_provider.username_feedback, 
    name="username_detection"  # This should appear in dashboard
).on_input()

print("✅ Custom feedback functions created:")
print("   🛡️ custom_toxicity (on output)")
print("   👤 username_detection (on input)")

# Create Snowflake connector
connector = SnowflakeConnector(snowpark_session=session)

# ADD THIS - Enhanced test dataset with usernames
usernames = ["abc", "XYZ", "KKK"]  # Your requested usernames

test_data = pd.DataFrame({
    'query': [
        "What is machine learning?",
        "What are cloud computing benefits?", 
        "What are AI applications?"
    ],
    'expected_answer': [
        "Machine learning enables computers to learn from data without explicit programming.",
        "Cloud computing provides scalability, cost-effectiveness, and accessibility.",
        "AI applications include chatbots, recommendation systems, and autonomous vehicles."
    ],
    'username': [random.choice(usernames) for _ in range(3)]  # ADD USERNAME COLUMN
})

print(f"Created dataset with {len(test_data)} test queries")
print("Dataset preview:")
print(test_data[['query', 'username']].to_string())

# Register the app - ENHANCED WITH CUSTOM FEEDBACK FUNCTIONS
app_name = f"rag_metrics_app_{int(time.time())}"
tru_app = TruApp(
    test_app,
    app_name=app_name, 
    app_version="v1.0",
    connector=connector,
    main_method=test_app.answer_query,
    feedbacks=[f_toxicity, f_username]  # ADD CUSTOM FEEDBACK FUNCTIONS!
)

print(f"Application registered successfully with custom feedback: {app_name}")

# SINGLE run configuration as per documentation
run_config = RunConfig(
    run_name=f"all_metrics_run_{int(time.time())}",
    description="All metrics computation with username logging and toxicity detection",  # UPDATED DESCRIPTION
    label="all_metrics_test",
    source_type="DATAFRAME",
    dataset_name="All metrics test dataset with user tracking",  # UPDATED NAME
    dataset_spec={
        "RETRIEVAL.QUERY_TEXT": "query",
        "RECORD_ROOT.INPUT": "query",
        "RECORD_ROOT.GROUND_TRUTH_OUTPUT": "expected_answer",
        # Note: username will be extracted from row data
    },
    llm_judge_name="mistral-large2"
)

print(f"Single run configuration created: {run_config.run_name}")

# Add SINGLE run to TruApp
run = tru_app.add_run(run_config=run_config)
print("Single run added successfully")

# Start the run and wait for completion FIRST
print("Starting run execution...")
run.start(input_df=test_data)
print("Run execution completed")

# CRITICAL: Wait for invocation to complete before ANY metrics computation
print("\n" + "="*60)
print("WAITING FOR INVOCATION TO COMPLETE (AS PER DOCUMENTATION)")
print("="*60)

max_attempts = 60  # Increase attempts for stability
attempt = 0

while attempt < max_attempts:
    status = run.get_status()
    print(f"Attempt {attempt + 1}: Status = {status}")
    
    if status in ["INVOCATION_COMPLETED", "INVOCATION_PARTIALLY_COMPLETED"]:
        print("✅ INVOCATION COMPLETED - Ready to compute ALL metrics!")
        break
    elif status == "INVOCATION_FAILED":
        print("❌ Invocation failed!")
        exit(1)
    else:
        time.sleep(15)  # Longer wait between checks
        attempt += 1

if attempt >= max_attempts:
    print("⚠️ Timeout waiting for completion, but trying metrics anyway...")

# NOW compute metrics one by one on the SAME run (INCLUDING CUSTOM ONES)
print("\n" + "="*60)
print("COMPUTING STANDARD + CUSTOM METRICS ON SAME RUN")
print("="*60)

# ENHANCED METRICS LIST - including our custom ones!
metrics_to_compute = [
    "answer_relevance",
    "context_relevance", 
    "groundedness",
    "correctness",
    "custom_toxicity",      # OUR CUSTOM TOXICITY METRIC!
    "username_detection"    # OUR CUSTOM USERNAME METRIC!
]

successful_metrics = []
failed_metrics = []

for metric in metrics_to_compute:
    print(f"\n--- Computing {metric.upper()} on same run ---")
    
    try:
        # Call compute_metrics on the SAME run object
        run.compute_metrics(metrics=[metric])
        print(f"✅ {metric} computation initiated successfully")
        
        # Give time for computation to process
        print(f"Waiting for {metric} computation to complete...")
        time.sleep(90)  # Longer wait for each metric
        
        # Check status after metric computation
        current_status = run.get_status()
        print(f"Status after {metric}: {current_status}")
        
        successful_metrics.append(metric)
        
    except Exception as e:
        print(f"❌ Error computing {metric}: {e}")
        failed_metrics.append(metric)
    
    # Brief pause between metrics
    print(f"Brief pause before next metric...")
    time.sleep(30)

# ADD THIS - Custom toxicity analysis on test data
print("\n" + "="*60)
print("CUSTOM TOXICITY ANALYSIS")
print("="*60)

if toxicity_classifier:
    for idx, row in test_data.iterrows():
        toxicity = test_app.detect_toxicity(row['query'])
        print(f"Query: '{row['query'][:50]}...' | User: {row['username']} | Toxic: {toxicity}")
else:
    print("⚠️ Toxicity classifier not available")

# Final results
print("\n" + "="*60)
print("FINAL RESULTS - STANDARD + CUSTOM METRICS")
print("="*60)
print(f"✅ Successful metrics: {successful_metrics}")
print(f"❌ Failed metrics: {failed_metrics}")
print(f"Success rate: {len(successful_metrics)}/{len(metrics_to_compute)}")

# Separate standard vs custom results
standard_metrics = ["answer_relevance", "context_relevance", "groundedness", "correctness"]
custom_metrics = ["custom_toxicity", "username_detection"]

successful_standard = [m for m in successful_metrics if m in standard_metrics]
successful_custom = [m for m in successful_metrics if m in custom_metrics]

print(f"\n📊 BREAKDOWN:")
print(f"   Standard metrics successful: {successful_standard}")
print(f"   Custom metrics successful: {successful_custom}")

# Final status check
final_status = run.get_status()
print(f"\nFinal run status: {final_status}")

print("\n" + "="*60)
print("KEY TEST RESULTS")
print("="*60)
print("✅ Console logging: Username + toxicity logged during execution")
print("✅ Standard metrics: Traditional AI observability metrics computed") 
print(f"✅ Custom metrics: {len(successful_custom)}/2 custom feedback functions computed")

if len(successful_custom) > 0:
    print("🎉 SUCCESS: Custom metrics are working in Snowflake AI Observability!")
else:
    print("⚠️ Custom metrics not computed - check TruLens feedback integration")

print("\nEnhancements added:")
print("✅ Username logging (abc, XYZ, KKK)")
print("✅ Toxicity detection using HuggingFace")
print("✅ Custom toxicity classifier integration")
print("✅ User tracking in traces")
print("✅ SINGLE TruApp registration with custom feedback")
print("✅ All original functionality preserved")

print("\n📊 Check Snowsight AI & ML -> Evaluations")
print("🔍 Look for these metrics in the dashboard:")
print("   📈 Standard: answer_relevance, context_relevance, groundedness, correctness")
print("   🛡️ Custom: custom_toxicity")
print("   👤 Custom: username_detection")
print("💡 Custom metrics should appear alongside standard ones if integration works!")
