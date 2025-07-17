import os
import pandas as pd
import time

# A. DEPENDENCY IMPORTS
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.session import Session
from snowflake.cortex import complete
from trulens_eval import TruCustomApp # NEW: Using TruCustomApp for automatic instrumentation
from trulens_eval.utils.threading import get_current_span # NEW: Utility to get the active span
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core.run import Run, RunConfig
from transformers import pipeline

# B. SETUP AND CONFIGURATION

# 1. Toxicity detection setup
try:
    print("Loading toxicity classifier model...")
    toxicity_classifier = pipeline("text-classification", model="s-nlp/roberta_toxicity_classifier")
    print("✅ Toxicity classifier loaded successfully.")
except Exception as e:
    print(f"⚠️ Could not load toxicity classifier: {e}")
    toxicity_classifier = None

# 2. Enable TruLens OpenTelemetry tracing
os.environ["TRULENS_OTEL_TRACING"] = "1"

# 3. Get Snowflake session
try:
    session = get_active_session()
    print("Using active Snowflake session.")
except Exception:
    print("Creating new Snowflake session from environment variables...")
    SNOWFLAKE_CONFIG = {
        "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
        "user": os.environ.get("SNOWFLAKE_USER"),
        "password": os.environ.get("SNOWFLAKE_PASSWORD"),
        "role": os.environ.get("SNOWFLAKE_ROLE"),
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
        "database": os.environ.get("SNOWFLAKE_DATABASE"),
        "schema": os.environ.get("SNOWFLAKE_SCHEMA")
    }
    if not all(SNOWFLAKE_CONFIG.values()):
        print("🛑 Error: Missing one or more Snowflake environment variables.")
        exit()
    session = Session.builder.configs(SNOWFLAKE_CONFIG).create()

print(f"✅ Snowflake session active. Current context: {session.get_current_database()}.{session.get_current_schema()}")

# C. HELPER FUNCTION FOR CUSTOM METRIC

def detect_toxicity_score(text: str) -> float:
    if toxicity_classifier is None or not text:
        return -1.0
    try:
        result = toxicity_classifier(text, top_k=None)
        toxic_score = next((item['score'] for item in result if item['label'] == 'TOXIC'), 0.0)
        return toxic_score
    except Exception as e:
        print(f"Error during toxicity detection: {e}")
        return -1.0

# D. RAG APPLICATION (CLEANED - NO MANUAL TRACING)

class RAGApplication:
    def __init__(self):
        self.model = "mistral-large2"
        # REMOVED: All manual tracer and OpenTelemetry code
        self.knowledge_base = {
            "machine learning": [
                "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
                "Common types include supervised learning, unsupervised learning, and reinforcement learning."
            ],
            "cloud computing": [
                "Cloud computing delivers computing services over the internet including storage, processing power, and applications.",
                "Major benefits include scalability, cost-effectiveness, and accessibility."
            ]
        }

    def retrieve_context(self, query: str) -> list:
        # This method is now plain Python. TruCustomApp will create the span.
        query_lower = query.lower()
        retrieved_contexts = []
        for topic, contexts in self.knowledge_base.items():
            if topic in query_lower:
                retrieved_contexts.extend(contexts)
                break
        if not retrieved_contexts:
            retrieved_contexts = ["Artificial intelligence enables machines to mimic human cognitive functions."]
        return retrieved_contexts

    def generate_completion(self, query: str, context_str: list) -> str:
        # This method is now plain Python. TruCustomApp will create the span.
        context_text = "\n".join([f"- {ctx}" for ctx in context_str])
        prompt = f"Based on the following context, answer the question.\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
        try:
            response = complete(self.model, prompt)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def answer_query(self, query: str) -> str:
        # This is the main method that TruLens will trace.
        context_str = self.retrieve_context(query)
        response = self.generate_completion(query, context_str)

        # --- KEY CHANGE: ADD ATTRIBUTE TO THE CURRENT SPAN ---
        span = get_current_span()
        if span:
            toxicity_score = detect_toxicity_score(response)
            span.set_attribute("response.toxicity_score", toxicity_score)
        # --------------------------------------------------------

        return response

# E. MAIN EXECUTION BLOCK

if __name__ == "__main__":
    # 1. Initialize the RAG application
    test_app = RAGApplication()

    # 2. Create Snowflake connector for TruLens
    connector = SnowflakeConnector(snowpark_session=session)

    # 3. Register the app with TruLens using TruCustomApp
    app_name = f"rag_full_trace_app_{int(time.time())}"
    # NEW: Using TruCustomApp will automatically instrument all methods
    tru_app = TruCustomApp(
        test_app,
        app_name=app_name,
        app_version="v1.0",
        connector=connector,
        main_method="answer_query" # Specify main method by name
    )
    print(f"✅ Application registered successfully: {app_name}")

    # 4. Create test dataset
    test_data = pd.DataFrame({
        'query': [
            "What is machine learning?",
            "Tell me about cloud computing benefits.",
            "You are a stupid machine."
        ]
    })
    print(f"✅ Created dataset with {len(test_data)} test queries.")

    # 5. Create a complete Run configuration
    run_config = RunConfig(
        run_name=f"full_trace_with_attributes_{int(time.time())}",
        description="Run to test full tracing with custom attributes",
        label="full_trace_test",
        source_type="DATAFRAME",
        dataset_name="Full Trace Test Data",
        dataset_spec={
            "RECORD_ROOT.INPUT": "query"
        }
    )
    print(f"✅ Run configuration created: {run_config.run_name}")

    # 6. Add and start the run
    run = tru_app.add_run(run_config=run_config)
    print("\n🚀 Starting run execution...")
    run.start(input_df=test_data)
    print("✅ Run execution completed.")

    print("\n---")
    print("🎉 End-to-end script finished successfully!")
    print("📊 Check Snowsight. You should now see the full trace with `retrieve_context` and `generate_completion` spans.")
    print(f"Your custom attribute 'response.toxicity_score' is still part of the trace data for the app '{app_name}'.")
    print("---\n")
