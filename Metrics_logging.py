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

    @instrument(span_type=SpanAttributes.SpanType.RETRIEVAL,
                attributes={
                    SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
                    SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return"
                })
    def retrieve_context(self, query: str) -> list:
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
    def generate_answer(self, query: str, context: list) -> str:
        context_str = "\n".join([f"- {ctx}" for ctx in context])
        
        prompt = f"""Given the context below, answer the question:
Context: {context_str}
Question: {query}
Answer:"""
        
        try:
            response = complete(self.model, prompt)
            return response
        except Exception as e:
            return f"Error: {str(e)}"

    @instrument(span_type=SpanAttributes.SpanType.RECORD_ROOT,
                attributes={
                    SpanAttributes.RECORD_ROOT.INPUT: "query",
                    SpanAttributes.RECORD_ROOT.OUTPUT: "return"
                })
    def answer_question(self, query: str) -> str:
        docs = self.retrieve_context(query)
        return self.generate_answer(query, docs)

# Initialize RAG app
rag_app = RAGApplication()

# Test basic functionality
print("Testing basic functionality...")
test_response = rag_app.answer_question("What is machine learning?")
print(f"Test response: {test_response[:100]}...")

# Create Snowflake connector
snowflake_conn = SnowflakeConnector(snowpark_session=session)

# Register app (Following Medium Article - NO main_method)
app_name = "fed_reserve_rag"
app_version = "v1"

tru_app = TruApp(rag_app,
                app_name=app_name,
                app_version=app_version,
                connector=snowflake_conn)

print("Application registered successfully")

# Create dataset (Following Medium Article Format)
test_data = pd.DataFrame({
    'QUERY': [
        "What is machine learning and how does it work?",
        "Explain cloud computing benefits for businesses",
        "What are the main applications of artificial intelligence?"
    ],
    'GROUND_TRUTH_RESPONSE': [
        "Machine learning is a subset of AI that enables computers to learn from data without explicit programming.",
        "Cloud computing provides scalability, cost-effectiveness, and accessibility for business operations.",
        "AI applications include chatbots, recommendation systems, autonomous vehicles, and medical diagnosis."
    ]
})

# Run configuration (Following Medium Article Exactly)
run_config = RunConfig(
    run_name="experiment_1_run",
    dataset_name="FOMC_DATA",
    description="Questions about the Federal Reserve FOMC meetings",
    label="baseline_run",
    source_type="DATAFRAME",
    dataset_spec={
        "input": "QUERY",
        "ground_truth_output": "GROUND_TRUTH_RESPONSE"
    }
)

# Add run
run = tru_app.add_run(run_config=run_config)

# Start run
print("Starting run...")
run.start(input_df=test_data)
print("Run completed")

# Compute metrics
print("Computing metrics...")
run.compute_metrics([
    "answer_relevance",
    "context_relevance", 
    "groundedness"
])

print("Metrics computation initiated")
print("Check Snowsight AI & ML -> Evaluations in a few minutes")

# Check results
try:
    trace_count = session.sql("""
        SELECT COUNT(*) as count 
        FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
        WHERE application_name = ?
    """, params=[app_name]).collect()
    
    if trace_count and trace_count[0]['COUNT'] > 0:
        print(f"SUCCESS: {trace_count[0]['COUNT']} traces found")
        
        span_types = session.sql("""
            SELECT span_type, COUNT(*) as count
            FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
            WHERE application_name = ?
            GROUP BY span_type
        """, params=[app_name]).collect()
        
        print("Span types:")
        for span in span_types:
            print(f"  {span['SPAN_TYPE']}: {span['COUNT']}")
    else:
        print("No traces found yet")
        
except Exception as e:
    print(f"Query failed: {e}")

print("\nSetup complete - check Snowsight for results!")
