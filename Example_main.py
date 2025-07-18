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

# Enable TruLens OpenTelemetry tracing
os.environ["TRULENS_OTEL_TRACING"] = "1"

# Get Snowflake session
session = get_active_session()

class SimpleRAGWithNativeAttributes:
    def __init__(self):
        self.model = "mistral-large2"
        
    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            "custom_username": "test_user_123",  # Try without "custom." prefix
            "complexity_score": 0.75,
            "user_id": "12345"
        }
    )
    def answer_query(self, query: str) -> str:
        """Try TruLens native attribute approach"""
        
        # Simple LLM call
        prompt = f"Answer briefly: {query}"
        try:
            response = complete(self.model, prompt)
            return response
        except Exception as e:
            return f"Error: {str(e)}"

# Initialize and test
test_app = SimpleRAGWithNativeAttributes()
connector = SnowflakeConnector(snowpark_session=session)
app_name = f"native_attributes_test_{int(time.time())}"

tru_app = TruApp(
    test_app,
    app_name=app_name,
    app_version="v1.0_native",
    connector=connector,
    main_method=test_app.answer_query
)

print(f"App registered: {app_name}")

# Minimal dataset
test_data = pd.DataFrame({
    'query': ["What is AI?"],
    'expected_answer': ["AI is artificial intelligence"]
})

# Run configuration
run_config = RunConfig(
    run_name=f"native_test_run_{int(time.time())}",
    description="Test TruLens native attributes",
    label="native_test",
    source_type="DATAFRAME", 
    dataset_name="Native test",
    dataset_spec={
        "RECORD_ROOT.INPUT": "query",
        "RECORD_ROOT.GROUND_TRUTH_OUTPUT": "expected_answer",
    },
    llm_judge_name="mistral-large2"
)

# Execute run
run = tru_app.add_run(run_config=run_config)
run.start(input_df=test_data)

# Wait and check
time.sleep(60)  # Wait longer for data propagation

print(f"✅ Test completed!")
print(f"🔍 Check these SQL queries with app_name = '{app_name}':")
print(f"")
print(f"1. SELECT * FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS")
print(f"   WHERE app_name = '{app_name}' ORDER BY timestamp DESC;")
print(f"")
print(f"2. SELECT record_attributes FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS") 
print(f"   WHERE app_name = '{app_name}' AND record_type = 'SPAN';")
print(f"")
print(f"3. Also check: SELECT * FROM SNOWFLAKE.TELEMETRY.EVENTS")
print(f"   WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '1 HOUR';")
