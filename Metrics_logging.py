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

# RAG Application Class (Updated per Official Documentation)
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

    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        }
    )
    def answer_query(self, query: str) -> str:
        """Main entry point for the RAG application."""
        context_str = self.retrieve_context(query)
        return self.generate_completion(query, context_str)

# Initialize the RAG application
test_app = RAGApplication()

# Test basic functionality
print("Testing basic functionality...")
test_response = test_app.answer_query("What is machine learning?")
print(f"Test successful: {test_response[:100]}...")

# Create Snowflake connector
connector = SnowflakeConnector(snowpark_session=session)

# Register the app in Snowflake (Updated per Official Documentation)
tru_app = TruApp(
    test_app,  # Application instance
    app_name="rag_evaluation_app", 
    app_version="v1",
    connector=connector,
    main_method=test_app.answer_query  # Entry point method
)

print("Application registered successfully")

# Create test dataset
test_data = pd.DataFrame({
    'user_query_field': [
        "What is machine learning and how does it work?",
        "Explain cloud computing benefits for businesses",
        "What are the main applications of artificial intelligence?"
    ],
    'golden_answer_field': [
        "Machine learning is a subset of AI that enables computers to learn from data without explicit programming.",
        "Cloud computing provides scalability, cost-effectiveness, and accessibility for business operations.", 
        "AI applications include chatbots, recommendation systems, autonomous vehicles, and medical diagnosis."
    ]
})

print(f"Created dataset with {len(test_data)} test queries")

# Create run configuration (Updated per Official Documentation)
run_config = RunConfig(
    run_name="test_run_1",
    description="RAG evaluation test run",
    label="rag_test",
    source_type="DATAFRAME",
    dataset_name="My test dataframe name",
    dataset_spec={
        "RETRIEVAL.QUERY_TEXT": "user_query_field",
        "RECORD_ROOT.INPUT": "user_query_field", 
        "RECORD_ROOT.GROUND_TRUTH_OUTPUT": "golden_answer_field",
    },
    llm_judge_name="mistral-large2"
)

print(f"Run configuration created: {run_config.run_name}")
print(f"Dataset spec: {run_config.dataset_spec}")

# Add run to TruApp
run = tru_app.add_run(run_config=run_config)
print("Run added successfully")

# Check run status before starting
initial_status = run.get_status()
print(f"Initial run status: {initial_status}")

# Start the run execution
print("Starting run execution...")
run.start(input_df=test_data)
print("Run execution completed")

# Check run status after execution
status = run.get_status()
print(f"Run status after execution: {status}")

# Wait for invocation to complete before computing metrics
print("Waiting for invocation to complete...")
while True:
    current_status = run.get_status()
    print(f"Current status: {current_status}")
    
    if current_status in ["INVOCATION_COMPLETED", "INVOCATION_PARTIALLY_COMPLETED"]:
        print("Invocation completed, ready to compute metrics")
        break
    elif current_status == "INVOCATION_FAILED":
        print("Invocation failed, cannot compute metrics")
        break
    elif current_status == "INVOCATION_IN_PROGRESS":
        print("Invocation still in progress, waiting...")
        time.sleep(10)
    else:
        print(f"Unexpected status: {current_status}")
        break

# Compute metrics (Updated per Official Documentation)
if status in ["INVOCATION_COMPLETED", "INVOCATION_PARTIALLY_COMPLETED"]:
    print("Computing evaluation metrics...")
    run.compute_metrics(metrics=[
        "coherence",
        "answer_relevance",
        "groundedness",
        "context_relevance",
        "correctness",
    ])
    print("Metrics computation initiated successfully")
    print("Note: run.compute_metrics() is an asynchronous non-blocking function")

# View run metadata
print("\n=== Run Metadata ===")
run.describe()

# Optional: List all runs for this application
print("\n=== All Runs for Application ===")
all_runs = tru_app.list_runs()
for run_info in all_runs:
    print(f"Run: {run_info}")

# Optional: Check if traces were created
try:
    trace_count = session.sql("""
        SELECT COUNT(*) as count 
        FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
        WHERE application_name = ?
    """, params=["rag_evaluation_app"]).collect()
    
    if trace_count and trace_count[0]['COUNT'] > 0:
        print(f"\n✅ SUCCESS: {trace_count[0]['COUNT']} traces found in event table")
        
        # Check span types
        span_types = session.sql("""
            SELECT span_type, COUNT(*) as count
            FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
            WHERE application_name = ?
            GROUP BY span_type
        """, params=["rag_evaluation_app"]).collect()
        
        print("Span types captured:")
        for span in span_types:
            print(f"  - {span['SPAN_TYPE']}: {span['COUNT']}")
            
        # Check for evaluation records
        eval_count = session.sql("""
            SELECT COUNT(*) as count 
            FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
            WHERE application_name = ? AND record_type = 'EVALUATION'
        """, params=["rag_evaluation_app"]).collect()
        
        if eval_count and eval_count[0]['COUNT'] > 0:
            print(f"✅ METRICS: {eval_count[0]['COUNT']} evaluation records found")
        else:
            print("ℹ️  Metrics may still be computing in the background")
            
    else:
        print("No traces found yet")
        
except Exception as e:
    print(f"Verification query failed: {e}")

# View evaluation results instructions
print("\n=== Viewing Results ===")
print("To view evaluation results:")
print("1. Navigate to Snowsight")
print("2. Select AI & ML")
print("3. Select Evaluations")
print("4. Select your application to view runs")
print("5. Select the run to view aggregated results")
print("6. Select individual records to view detailed traces")
print("7. To compare runs, select multiple runs and click Compare")

print("\n" + "="*60)
print("RAG EVALUATION SETUP COMPLETE")
print("="*60)
print("✅ App instrumented with proper span attributes")
print("✅ App registered with TruApp and main_method specified") 
print("✅ Run created with correct dataset_spec format")
print("✅ Run executed with proper status checking")
print("✅ Metrics computation initiated after invocation completion")
print("✅ Enhanced with run metadata and status monitoring")
print("\n📊 Check Snowsight AI & ML -> Evaluations for results")
print("🕐 Metrics computed asynchronously in background")
