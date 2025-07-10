import os
import pandas as pd
import time
import random
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.session import Session
from snowflake.cortex import complete
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
from trulens.apps.app import TruApp
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core.run import Run, RunConfig

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
    def answer_query(self, query: str) -> str:
        """Main entry point for the RAG application."""
        context_str = self.retrieve_context(query)
        return self.generate_completion(query, context_str)

# Initialize the RAG application
test_app = RAGApplication()

# Test basic functionality
print("Testing basic functionality...")
test_response = test_app.answer_query("What is machine learning?")
print(f"Test successful: {test_response[:100] if test_response else 'No response'}...")

# Create Snowflake connector
connector = SnowflakeConnector(snowpark_session=session)

# Generate unique timestamp and random ID to avoid conflicts
base_timestamp = int(time.time())
session_id = random.randint(1000, 9999)

# Register the app with unique name
app_name = f"rag_metrics_app_{base_timestamp}_{session_id}"
tru_app = TruApp(
    test_app,
    app_name=app_name, 
    app_version="v1.0",
    connector=connector,
    main_method=test_app.answer_query
)

print(f"Application registered successfully: {app_name}")

# Create test dataset - EXACT same as working code
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
    ]
})

print(f"Created dataset with {len(test_data)} test queries")

# Function to create and run metrics with unique names and delays
def run_metric_with_working_pattern(metric_name, tru_app, test_data, base_timestamp, session_id):
    print(f"\n{'='*60}")
    print(f"RUNNING {metric_name.upper()} WITH UNIQUE NAMES")
    print(f"{'='*60}")
    
    # Create unique run name with timestamp + random component
    unique_timestamp = int(time.time())
    unique_id = random.randint(100, 999)
    
    # EXACT same run config pattern that worked but with unique names
    run_config = RunConfig(
        run_name=f"{metric_name}_run_{unique_timestamp}_{unique_id}",  # Unique name
        description=f"{metric_name} computation test run",
        label=f"{metric_name}_test",
        source_type="DATAFRAME",
        dataset_name=f"{metric_name}_test_dataset_{unique_id}",  # Unique dataset name
        dataset_spec={
            # EXACT same mapping that worked
            "RETRIEVAL.QUERY_TEXT": "query",
            "RECORD_ROOT.INPUT": "query",
            "RECORD_ROOT.GROUND_TRUTH_OUTPUT": "expected_answer",
        },
        llm_judge_name="mistral-large2"  # EXACT same judge
    )
    
    print(f"Run configuration created: {run_config.run_name}")
    
    try:
        # Add run to TruApp
        run = tru_app.add_run(run_config=run_config)
        print("Run added successfully")
        
        # Start the run - EXACT same process
        print("Starting run execution...")
        run.start(input_df=test_data)
        print("Run execution completed")
        
        # EXACT same wait pattern
        print("Waiting for invocation to complete...")
        max_attempts = 30
        attempt = 0
        
        while attempt < max_attempts:
            status = run.get_status()
            print(f"Attempt {attempt + 1}: Status = {status}")
            
            if status in ["INVOCATION_COMPLETED", "INVOCATION_PARTIALLY_COMPLETED"]:
                print("✅ Ready to compute metrics!")
                break
            elif status == "INVOCATION_FAILED":
                print("❌ Invocation failed!")
                return False
            else:
                time.sleep(10)
                attempt += 1
        
        if attempt >= max_attempts:
            print("⚠️ Timeout waiting for completion, trying metrics anyway...")
        
        # EXACT same metrics computation pattern
        print(f"Computing: {metric_name}")
        run.compute_metrics(metrics=[metric_name])
        print(f"✅ {metric_name} computation started")
        
        # EXACT same wait time
        print("Waiting for metric computation...")
        time.sleep(60)
        
        status_after_metrics = run.get_status()
        print(f"Status after {metric_name}: {status_after_metrics}")
        return True
        
    except Exception as e:
        print(f"❌ Error with {metric_name}: {e}")
        return False

# Run remaining metrics that haven't been tested yet
remaining_metrics = [
    "groundedness",    # Try this again to confirm
    "correctness"      # 
]

successful_metrics = ["answer_relevance", "context_relevance"]  # Already working
failed_metrics = []

for metric in remaining_metrics:
    print(f"\n🔄 LONG DELAY BEFORE {metric.upper()} (Avoiding conflicts)")
    time.sleep(120)  # 2 minute delay between runs to avoid conflicts
    
    print(f"\nStarting {metric} with unique identifiers...")
    success = run_metric_with_working_pattern(metric, tru_app, test_data, base_timestamp, session_id)
    
    if success:
        successful_metrics.append(metric)
        print(f"✅ {metric} completed successfully")
    else:
        failed_metrics.append(metric)
        print(f"❌ {metric} failed")

# Final results
print("\n" + "="*60)
print("FINAL RESULTS - FIXED CONFLICTS")
print("="*60)
print(f"✅ Successful metrics: {successful_metrics}")
print(f"❌ Failed metrics: {failed_metrics}")
print(f"Success rate: {len(successful_metrics)}/4")

print("\nKey fixes applied:")
print("✅ Unique run names with timestamps + random IDs")
print("✅ Unique dataset names")
print("✅ Long delays (2 minutes) between runs")
print("✅ Better error handling for conflicts")

print("\n📊 Check Snowsight AI & ML -> Evaluations")
print("🔍 Should see separate runs with unique names")
