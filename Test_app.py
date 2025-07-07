
import os
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import complete
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
from trulens.apps.app import TruApp
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core.run import Run, RunConfig
import pandas as pd

# Enable TruLens OpenTelemetry tracing
os.environ["TRULENS_OTEL_TRACING"] = "1"

# Get Snowflake session
session = get_active_session()

# Simple LLM Application with Cortex Complete
class SimpleLLMApp:
    def __init__(self):
        self.model = "mistral-large2"  # or use "llama3.1-70b", "claude-3.5-sonnet", etc.
    
    @instrument(
        span_type=SpanAttributes.SpanType.GENERATION,
        attributes={
            SpanAttributes.GENERATION.INPUT_MESSAGES: "query",
            SpanAttributes.GENERATION.OUTPUT_MESSAGES: "return",
        }
    )
    def generate_response(self, query: str) -> str:
        """Generate response using Cortex Complete"""
        prompt = f"""
        You are a helpful assistant. Answer the following question clearly and concisely.
        
        Question: {query}
        
        Answer:
        """
        
        try:
            # Using Cortex Complete function
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
    def query(self, query: str) -> str:
        """Main query method - this is the entry point for evaluation"""
        return self.generate_response(query)

# Initialize the application
llm_app = SimpleLLMApp()

# Test the application manually first
print("Testing the application manually:")
test_response = llm_app.query("What is artificial intelligence?")
print(f"Response: {test_response}")

# Set up TruLens connector for Snowflake
tru_snowflake_connector = SnowflakeConnector(snowpark_session=session)

# Register the application with TruLens
app_name = "simple_llm_test"
app_version = "v1"

tru_llm_app = TruApp(
    llm_app,
    app_name=app_name,
    app_version=app_version,
    connector=tru_snowflake_connector
)

# Create a simple test dataset
test_data = pd.DataFrame({
    'QUERY': [
        "What is artificial intelligence?",
        "Explain machine learning in simple terms",
        "What are the benefits of cloud computing?",
        "How does natural language processing work?",
        "What is the difference between AI and ML?"
    ],
    'EXPECTED_ANSWER': [
        "AI is computer technology that can perform tasks typically requiring human intelligence",
        "ML is a method of teaching computers to learn patterns from data",
        "Cloud computing offers scalability, cost-effectiveness, and accessibility",
        "NLP enables computers to understand and process human language",
        "AI is broader field, ML is a subset of AI focused on learning from data"
    ]
})

# Upload test dataset to Snowflake
test_data.to_snowflake("TEST_QUERIES", if_exists="replace", index=False)

# Configure the evaluation run
run_name = "simple_llm_evaluation_run"
run_config = RunConfig(
    run_name=run_name,
    dataset_name="TEST_QUERIES",
    description="Simple LLM evaluation test",
    label="llm_test",
    source_type="TABLE",
    dataset_spec={
        "input": "QUERY",
        "ground_truth_output": "EXPECTED_ANSWER",
    },
)

# Add the run to TruLens
run: Run = tru_llm_app.add_run(run_config=run_config)

print(f"Starting evaluation run: {run_name}")

# Start the evaluation run
run.start()

print("Run completed. Computing metrics...")

# Compute evaluation metrics
run.compute_metrics([
    "answer_relevance",
    "groundedness",
])

print("Evaluation completed!")
print(f"Check results in Snowsight: AI & ML > Evaluations > {app_name}")

# Optional: Check external agent was created
session.sql("SHOW EXTERNAL AGENTS").show()

# Optional: Check if data was ingested into event table
session.sql("""
    SELECT COUNT(*) as event_count 
    FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
    WHERE application_name = ?
""", params=[app_name]).show()

print("\n=== Setup Verification ===")
print("1. Check if external agent exists: SHOW EXTERNAL AGENTS;")
print("2. Check observability data: SELECT * FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS;")
print("3. View results in Snowsight: AI & ML → Evaluations")
