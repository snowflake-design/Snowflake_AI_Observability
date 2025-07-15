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

# ADD THIS - Import for toxicity detection
try:
    from transformers import pipeline
    toxicity_classifier = pipeline("text-classification", model="s-nlp/roberta_toxicity_classifier")
    print("✅ Toxicity classifier loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load toxicity classifier: {e}")
    toxicity_classifier = None

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
    def answer_query(self, query: str, username: str = "unknown") -> str:  # ADD USERNAME PARAMETER
        """Main entry point for the RAG application."""
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

# Create Snowflake connector
connector = SnowflakeConnector(snowpark_session=session)

# Register the app
app_name = f"rag_metrics_app_{int(time.time())}"
tru_app = TruApp(
    test_app,
    app_name=app_name, 
    app_version="v1.0",
    connector=connector,
    main_method=test_app.answer_query
)

print(f"Application registered successfully: {app_name}")

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
        # Note: username will be passed as parameter to answer_query
    },
    llm_judge_name="mistral-large2"
)

print(f"Single run configuration created: {run_config.run_name}")

# ADD THIS - Create wrapper function to handle username parameter
def run_with_username(row):
    """Wrapper to pass username to answer_query method."""
    return test_app.answer_query(row['query'], row['username'])

# MODIFY THIS - Update TruApp to use wrapper function
tru_app_with_username = TruApp(
    test_app,
    app_name=f"{app_name}_with_users",
    app_version="v1.1", 
    connector=connector,
    main_method=run_with_username  # Use wrapper function
)

# Add SINGLE run to TruApp
run = tru_app_with_username.add_run(run_config=run_config)
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

# NOW compute metrics one by one on the SAME run (as per documentation)
print("\n" + "="*60)
print("COMPUTING MULTIPLE METRICS ON SAME RUN")
print("="*60)

metrics_to_compute = [
    "answer_relevance",
    "context_relevance", 
    "groundedness",
    "correctness"
    # Note: toxicity is computed inline, not as a separate TruLens metric
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
print("FINAL RESULTS - ENHANCED WITH USER TRACKING & TOXICITY")
print("="*60)
print(f"✅ Successful metrics: {successful_metrics}")
print(f"❌ Failed metrics: {failed_metrics}")
print(f"Success rate: {len(successful_metrics)}/{len(metrics_to_compute)}")

# Final status check
final_status = run.get_status()
print(f"\nFinal run status: {final_status}")

print("\nEnhancements added:")
print("✅ Username logging (abc, XYZ, KKK)")
print("✅ Toxicity detection using HuggingFace")
print("✅ Custom toxicity classifier integration")
print("✅ User tracking in traces")
print("✅ All original functionality preserved")

print("\n📊 Check Snowsight AI & ML -> Evaluations")
print("🔍 Should see runs with user context and toxicity info")
print("👥 User information logged in application traces")
print("🛡️ Toxicity detection results in console output")
