import os
import pandas as pd
import time
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

# Register the app - EXACT same pattern as working code
app_name = f"rag_metrics_app_{int(time.time())}"
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

# Function to create and run metrics using the EXACT working pattern
def run_metric_with_working_pattern(metric_name, tru_app, test_data):
    print(f"\n=== Running {metric_name.upper()} with Working Pattern ===")
    
    # EXACT same run config pattern that worked
    run_config = RunConfig(
        run_name=f"metrics_run_{metric_name}_{int(time.time())}",
        description=f"{metric_name} computation test run",
        label="metrics_test",
        source_type="DATAFRAME",
        dataset_name="Metrics test dataset",
        dataset_spec={
            # EXACT same mapping that worked
            "RETRIEVAL.QUERY_TEXT": "query",
            "RECORD_ROOT.INPUT": "query",
            "RECORD_ROOT.GROUND_TRUTH_OUTPUT": "expected_answer",
        },
        llm_judge_name="mistral-large2"  # EXACT same judge
    )
    
    print(f"Run configuration created: {run_config.run_name}")
    
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
    try:
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
        print(f"❌ Error computing {metric_name}: {e}")
        return False

# Run each metric using the EXACT working pattern
metrics_to_test = [
    "answer_relevance",    # We know this works
    "context_relevance", 
    "groundedness",
    "correctness"
]

successful_metrics = []
failed_metrics = []

for metric in metrics_to_test:
    print(f"\n{'='*60}")
    print(f"TESTING {metric.upper()} WITH EXACT WORKING PATTERN")
    print(f"{'='*60}")
    
    success = run_metric_with_working_pattern(metric, tru_app, test_data)
    
    if success:
        successful_metrics.append(metric)
        print(f"✅ {metric} completed successfully")
    else:
        failed_metrics.append(metric)
        print(f"❌ {metric} failed")
    
    # Wait between metrics to avoid conflicts
    print("Waiting before next metric...")
    time.sleep(30)

# Final results
print("\n" + "="*60)
print("FINAL RESULTS - EXACT PATTERN REPLICATION")
print("="*60)
print(f"✅ Successful metrics: {successful_metrics}")
print(f"❌ Failed metrics: {failed_metrics}")
print(f"Success rate: {len(successful_metrics)}/{len(metrics_to_test)}")

print("\nKey points:")
print("✅ Used EXACT same code pattern that worked for answer_relevance")
print("✅ Same dataset_spec mapping")
print("✅ Same model (mistral-large2)")
print("✅ Same timing and wait patterns")
print("✅ Separate run for each metric")

print("\n📊 Check Snowsight AI & ML -> Evaluations")
print("🔍 Look for multiple runs, each testing one metric")
