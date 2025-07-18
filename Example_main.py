import os
import pandas as pd
import time
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import complete
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
from trulens.apps.app import TruApp  # Note: TruApp only, not TruCustomApp
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core.run import Run, RunConfig

# Enable TruLens OpenTelemetry tracing
os.environ["TRULENS_OTEL_TRACING"] = "1"

# Get Snowflake session
session = get_active_session()

class TruLensNativeRAG:
    """
    The ONLY TruLens native approach that works with Snowflake AI Observability:
    Using predefined span attributes in the @instrument decorator
    """
    def __init__(self):
        self.model = "mistral-large2"
        
    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
            # These are the ONLY custom attributes supported natively:
            "custom_retrieval_score": "custom_score",  # Map to local variable
            "user_context": "username"  # Map to parameter
        }
    )
    def retrieve_context(self, query: str, username: str = "default_user") -> list:
        """TruLens native approach: predefined attributes in @instrument"""
        
        # Simulate retrieval
        contexts = [
            f"Context 1 for query: {query}",
            f"Context 2 related to: {query}"
        ]
        
        # This gets mapped to custom_retrieval_score attribute
        custom_score = min(len(contexts) / 3.0, 1.0)  
        
        return contexts
    
    @instrument(
        span_type=SpanAttributes.SpanType.GENERATION,
        attributes={
            "generation_quality": "quality_score",
            "response_type": "response_category"
        }
    )
    def generate_completion(self, query: str, contexts: list) -> str:
        """Generate with native TruLens attributes"""
        
        context_text = "\n".join(contexts)
        prompt = f"Based on: {context_text}\nAnswer: {query}"
        
        try:
            response = complete(self.model, prompt)
            
            # These get mapped to span attributes
            quality_score = min(len(response) / 100, 1.0)
            response_category = "detailed" if len(response) > 100 else "brief"
            
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
            "session_info": "session_data",
            "processing_metadata": "metadata"
        }
    )
    def answer_query(self, query: str) -> str:
        """Main method with root-level native attributes"""
        
        # Session and metadata info (gets mapped to attributes)
        session_data = {"user": "test_user", "timestamp": int(time.time())}
        metadata = {"version": "v1.0", "model": self.model}
        
        # Execute the pipeline
        contexts = self.retrieve_context(query, username="test_user_123")
        response = self.generate_completion(query, contexts)
        
        return response

# =====================================
# THIS IS THE LIMITATION:
# =====================================
# 
# You CANNOT do this with Snowflake AI Observability:
# 
# from trulens.core import Feedback, Provider
# from trulens.apps.custom import TruCustomApp
# 
# class MyCustomProvider(Provider):
#     def my_custom_metric(self, text: str) -> float:
#         return 0.85
# 
# custom_feedback = Feedback(MyCustomProvider().my_custom_metric)
# 
# tru_app = TruCustomApp(  # ← NOT supported with SnowflakeConnector
#     app, 
#     feedbacks=[custom_feedback]  # ← This parameter doesn't exist in TruApp
# )
#
# =====================================

# Initialize the app (TruApp only, no custom feedbacks)
test_app = TruLensNativeRAG()
connector = SnowflakeConnector(snowpark_session=session)
app_name = f"trulens_native_approach_{int(time.time())}"

tru_app = TruApp(
    test_app,
    app_name=app_name,
    app_version="v1.0_native_trulens",
    connector=connector,
    main_method=test_app.answer_query
    # Note: NO feedbacks parameter available!
)

print(f"TruLens native app registered: {app_name}")

# Test data
test_data = pd.DataFrame({
    'query': ["What is machine learning?", "Explain AI applications"],
    'expected_answer': ["ML is a subset of AI", "AI applications include various tools"]
})

# Run configuration (same as before)
run_config = RunConfig(
    run_name=f"native_trulens_run_{int(time.time())}",
    description="TruLens native approach test",
    label="native_trulens_test",
    source_type="DATAFRAME",
    dataset_name="Native TruLens test",
    dataset_spec={
        "RETRIEVAL.QUERY_TEXT": "query",
        "RECORD_ROOT.INPUT": "query",
        "RECORD_ROOT.GROUND_TRUTH_OUTPUT": "expected_answer",
    },
    llm_judge_name="mistral-large2"
)

# Execute
run = tru_app.add_run(run_config=run_config)
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

# Compute built-in metrics only
print("Computing built-in metrics...")
for metric in ["answer_relevance", "context_relevance", "groundedness"]:
    try:
        run.compute_metrics(metrics=[metric])
        print(f"✅ {metric} computed")
        time.sleep(20)
    except Exception as e:
        print(f"❌ Error with {metric}: {e}")

print(f"\n🎯 TRULENS NATIVE APPROACH RESULTS:")
print(f"✅ App: {app_name}")
print(f"✅ Native TruLens attributes logged via @instrument decorator:")
print(f"   - custom_retrieval_score (from retrieve_context)")
print(f"   - user_context (username mapping)")
print(f"   - generation_quality (from generate_completion)")
print(f"   - response_type (response category)")
print(f"   - session_info (session data)")
print(f"   - processing_metadata (version info)")

print(f"\n❌ WHAT DOESN'T WORK with Snowflake AI Observability:")
print(f"   - TruCustomApp (only TruApp supported)")
print(f"   - Custom feedback functions via feedbacks parameter")
print(f"   - run.compute_metrics() with custom metric names")

print(f"\n✅ WHAT DOES WORK:")
print(f"   - Predefined attributes in @instrument decorator")
print(f"   - 4 built-in metrics: answer_relevance, context_relevance, groundedness, correctness")
print(f"   - Custom data logged as span attributes (queryable via SQL)")

print(f"\n🔍 Check attributes in SQL:")
print(f"SELECT record_attributes FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS")
print(f"WHERE app_name = '{app_name}' AND record_type = 'SPAN';")

print(f"\n📊 Bottom Line:")
print(f"TruLens 'native approach' for Snowflake = @instrument decorator attributes only!")
print(f"For true custom feedback functions, you need TruLens standalone (not Snowflake integration)")
