import os
import pandas as pd
import time
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import complete
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
from trulens.apps.app import TruApp
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core.run import Run, RunConfig

# For custom span attributes
from opentelemetry import trace

# Enable TruLens OpenTelemetry tracing
os.environ["TRULENS_OTEL_TRACING"] = "1"

# Get Snowflake session
session = get_active_session()
print(f"Current context: {session.get_current_database()}.{session.get_current_schema()}")

class SimpleRAG:
    def __init__(self):
        self.model = "mistral-large2"
        
    @instrument(span_type=SpanAttributes.SpanType.RECORD_ROOT)
    def answer_query(self, query: str) -> str:
        """Simple RAG with custom attributes"""
        # Get current span for custom attributes
        current_span = trace.get_current_span()
        
        # Add custom attributes: username and custom metric
        current_span.set_attribute("custom.username", "test_user_123")
        current_span.set_attribute("custom.query_complexity_score", 0.75)
        
        # Simple LLM call
        prompt = f"Answer this question briefly: {query}"
        try:
            response = complete(self.model, prompt)
            
            # Add response length as custom metric
            current_span.set_attribute("custom.response_length", len(response))
            
            return response
        except Exception as e:
            current_span.set_attribute("custom.error", str(e))
            return f"Error: {str(e)}"

# Initialize app
test_app = SimpleRAG()

# Test functionality
print("Testing basic functionality...")
test_response = test_app.answer_query("What is AI?")
print(f"Test response: {test_response[:50]}...")

# Create Snowflake connector and register app
connector = SnowflakeConnector(snowpark_session=session)
app_name = f"minimal_test_app_{int(time.time())}"

tru_app = TruApp(
    test_app,
    app_name=app_name, 
    app_version="v1.0_minimal",
    connector=connector,
    main_method=test_app.answer_query
)

print(f"App registered: {app_name}")

# Minimal test dataset
test_data = pd.DataFrame({
    'query': ["What is AI?", "What is machine learning?"],
    'expected_answer': ["AI is artificial intelligence", "ML is a subset of AI"]
})

# Run configuration (keeping same as your original)
run_config = RunConfig(
    run_name=f"minimal_test_run_{int(time.time())}",
    description="Minimal test with custom attributes",
    label="minimal_test",
    source_type="DATAFRAME",
    dataset_name="Minimal test dataset",
    dataset_spec={
        "RECORD_ROOT.INPUT": "query",
        "RECORD_ROOT.GROUND_TRUTH_OUTPUT": "expected_answer",
    },
    llm_judge_name="mistral-large2"
)

# Add and start run
run = tru_app.add_run(run_config=run_config)
print("Starting run...")
run.start(input_df=test_data)

# Wait for completion
max_attempts = 30
for attempt in range(max_attempts):
    status = run.get_status()
    print(f"Attempt {attempt + 1}: Status = {status}")
    
    if status in ["INVOCATION_COMPLETED", "INVOCATION_PARTIALLY_COMPLETED"]:
        print("✅ INVOCATION COMPLETED!")
        break
    elif status == "INVOCATION_FAILED":
        print("❌ Invocation failed!")
        break
    else:
        time.sleep(10)

# Compute one metric
print("Computing answer_relevance metric...")
try:
    run.compute_metrics(metrics=["answer_relevance"])
    print("✅ Metric computation initiated")
    time.sleep(30)  # Wait for completion
except Exception as e:
    print(f"❌ Error computing metric: {e}")

print(f"\n🎯 RESULTS:")
print(f"✅ App: {app_name}")
print(f"✅ Custom attributes logged:")
print(f"   - custom.username: test_user_123")
print(f"   - custom.query_complexity_score: 0.75")
print(f"   - custom.response_length: [actual response length]")

print(f"\n📊 Check in Snowflake:")
print(f"   AI & ML -> Evaluations -> {app_name}")
print(f"\n🔍 Query custom attributes with SQL:")
print(f"   SELECT * FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS")
print(f"   WHERE app_name = '{app_name}';")

print(f"\n✅ Test completed! Custom attributes should be in Snowflake.")
