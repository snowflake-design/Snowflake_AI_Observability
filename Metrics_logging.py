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

# Register the app - CRITICAL: Use unique app name
app_name = f"rag_metrics_app_{int(time.time())}"
tru_app = TruApp(
    test_app,
    app_name=app_name, 
    app_version="v1.0",
    connector=connector,
    main_method=test_app.answer_query
)

print(f"Application registered successfully: {app_name}")

# Create test dataset - CRITICAL: Simple, clear data
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

# CRITICAL FIX: Proper dataset_spec mapping
run_config = RunConfig(
    run_name=f"metrics_run_{int(time.time())}",
    description="Metrics computation test run",
    label="metrics_test",
    source_type="DATAFRAME",
    dataset_name="Metrics test dataset",
    dataset_spec={
        # Map dataset columns to span attributes - THIS IS THE KEY FIX
        "RETRIEVAL.QUERY_TEXT": "query",           # Maps to query column
        "RECORD_ROOT.INPUT": "query",              # Maps to query column  
        "RECORD_ROOT.GROUND_TRUTH_OUTPUT": "expected_answer",  # Maps to expected_answer column
    },
    llm_judge_name="mistral-large2"
)

print(f"Run configuration created: {run_config.run_name}")
print(f"Dataset spec mapping: {run_config.dataset_spec}")

# Add run to TruApp
run = tru_app.add_run(run_config=run_config)
print("Run added successfully")

# Start the run
print("Starting run execution...")
run.start(input_df=test_data)
print("Run execution completed")

# CRITICAL: Wait for proper status
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
        exit(1)
    else:
        time.sleep(10)
        attempt += 1

if attempt >= max_attempts:
    print("⚠️ Timeout waiting for completion, trying metrics anyway...")

# CRITICAL FIX: Start with basic metrics only
print("\n=== Computing Basic Metrics ===")
basic_metrics = ["answer_relevance"]

try:
    print(f"Computing: {basic_metrics}")
    run.compute_metrics(metrics=basic_metrics)
    print("✅ Basic metrics computation started")
    
    # Wait and check for results
    print("Waiting for metric computation...")
    time.sleep(60)  # Give more time for computation
    
    # Check if metrics were computed
    try:
        status_after_metrics = run.get_status()
        print(f"Status after metrics: {status_after_metrics}")
        
        # Try to add more metrics if first one succeeded
        print("\n=== Adding More Metrics ===")
        additional_metrics = ["context_relevance"]
        run.compute_metrics(metrics=additional_metrics)
        print("✅ Additional metrics computation started")
        
    except Exception as e:
        print(f"Error checking status or adding metrics: {e}")
        
except Exception as e:
    print(f"❌ Error computing basic metrics: {e}")
    print("Common causes:")
    print("1. Insufficient permissions for Cortex LLM access")
    print("2. Model 'mistral-large2' not available")
    print("3. Dataset mapping incorrect")
    print("4. Missing required span attributes")

# Alternative: Try with different LLM judge
print("\n=== Trying Alternative LLM Judge ===")
try:
    # Create new run with different judge
    alt_run_config = RunConfig(
        run_name=f"alt_metrics_run_{int(time.time())}",
        description="Alternative LLM judge test",
        label="alt_metrics_test", 
        source_type="DATAFRAME",
        dataset_name="Alt metrics test dataset",
        dataset_spec={
            "RETRIEVAL.QUERY_TEXT": "query",
            "RECORD_ROOT.INPUT": "query",
            "RECORD_ROOT.GROUND_TRUTH_OUTPUT": "expected_answer",
        },
        llm_judge_name="llama3.1-70b"  # Try default judge
    )
    
    alt_run = tru_app.add_run(run_config=alt_run_config)
    alt_run.start(input_df=test_data)
    
    # Wait for completion
    time.sleep(30)
    alt_status = alt_run.get_status()
    
    if alt_status in ["INVOCATION_COMPLETED", "INVOCATION_PARTIALLY_COMPLETED"]:
        print("Trying with default LLM judge...")
        alt_run.compute_metrics(metrics=["answer_relevance"])
        print("✅ Alternative metrics started")
    
except Exception as e:
    print(f"Alternative approach failed: {e}")

# Final status check
print("\n=== Final Results ===")
try:
    final_status = run.get_status()
    print(f"Final run status: {final_status}")
    
    # List all runs
    all_runs = tru_app.list_runs()
    print(f"Total runs created: {len(all_runs)}")
    
except Exception as e:
    print(f"Error in final check: {e}")

print("\n" + "="*60)
print("METRICS TROUBLESHOOTING COMPLETE")
print("="*60)
print("Key fixes applied:")
print("✅ Proper dataset_spec column mapping")
print("✅ Unique app and run names")
print("✅ Simple, clear test data")
print("✅ Basic metrics first approach")
print("✅ Alternative LLM judge attempt")
print("\n📊 Check Snowsight AI & ML -> Evaluations")
print("🔍 If still no metrics, check:")
print("   - CORTEX_USER role permissions")
print("   - LLM model availability") 
print("   - Span attribute capture in traces")
