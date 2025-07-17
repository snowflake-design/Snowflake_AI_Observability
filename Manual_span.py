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
from opentelemetry import trace

# Enable TruLens OpenTelemetry tracing
os.environ["TRULENS_OTEL_TRACING"] = "1"

# Get Snowflake session
session = get_active_session()

class SimpleRAG:
    def __init__(self):
        self.current_username = "anonymous"
        self.knowledge_base = {
            "ai": "Artificial Intelligence is the simulation of human intelligence by machines.",
            "ml": "Machine Learning is a subset of AI that enables computers to learn from data.",
            "python": "Python is a high-level programming language used for AI and data science."
        }
    
    def set_username(self, username: str):
        """Set current username for tracking"""
        self.current_username = username
        print(f"Username set to: {username}")
    
    # ✅ RETRIEVAL span - HAS attributes: QUERY_TEXT, RETRIEVED_CONTEXTS
    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        }
    )
    def retrieve_context(self, query: str) -> list:
        """Retrieve context from knowledge base"""
        # Add custom attributes via OpenTelemetry (this is the ONLY way to add custom data)
        current_span = trace.get_current_span()
        if current_span:
            # Username tracking - stored as custom span attribute
            current_span.set_attribute("user.name", self.current_username)
            current_span.set_attribute("user.query_timestamp", int(time.time()))
            
            # Custom metrics as span attributes (not real metrics!)
            query_length = len(query)
            current_span.set_attribute("custom.query_length", query_length)
            current_span.set_attribute("custom.query_complexity", "high" if query_length > 20 else "low")
        
        # Simple keyword matching
        query_lower = query.lower()
        for key, value in self.knowledge_base.items():
            if key in query_lower:
                result = [value]
                
                # Add more custom attributes
                if current_span:
                    current_span.set_attribute("custom.topic_matched", key)
                    current_span.set_attribute("custom.context_count", len(result))
                
                return result
        
        # Default fallback
        result = [self.knowledge_base["ai"]]
        if current_span:
            current_span.set_attribute("custom.topic_matched", "ai")
            current_span.set_attribute("custom.fallback_used", True)
        
        return result
    
    # ❌ GENERATION span - NO attributes allowed (as per documentation)
    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate_completion(self, query: str, context_str: list) -> str:
        """Generate response using LLM"""
        # Can only add custom attributes via OpenTelemetry API
        current_span = trace.get_current_span()
        if current_span:
            # Username tracking in generation span
            current_span.set_attribute("user.name", self.current_username)
            current_span.set_attribute("generation.model", "mistral-large2")
        
        context_text = "\n".join(context_str)
        prompt = f"Context: {context_text}\nQuestion: {query}\nAnswer:"
        
        response = complete("mistral-large2", prompt)
        
        # Add custom metrics as span attributes
        if current_span:
            # Simple toxicity check (heuristic since no HuggingFace)
            toxic_words = ["hate", "bad", "terrible", "awful"]
            toxicity_score = sum(1 for word in toxic_words if word.lower() in response.lower())
            current_span.set_attribute("custom.toxicity_score", toxicity_score)
            current_span.set_attribute("custom.response_length", len(response))
        
        return response
    
    # ✅ RECORD_ROOT span - HAS attributes: INPUT, OUTPUT, GROUND_TRUTH_OUTPUT
    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        }
    )
    def answer_query(self, query: str) -> str:
        """Main RAG pipeline entry point"""
        # Add custom attributes for username tracking
        current_span = trace.get_current_span()
        if current_span:
            current_span.set_attribute("user.name", self.current_username)
            current_span.set_attribute("user.session_id", f"session_{int(time.time())}")
            current_span.set_attribute("pipeline.version", "v1.0")
        
        context_str = self.retrieve_context(query)
        response = self.generate_completion(query, context_str)
        
        # Add final custom metrics
        if current_span:
            current_span.set_attribute("custom.pipeline_completed", True)
            current_span.set_attribute("custom.total_processing_time", time.time())
        
        return response

# Initialize RAG
rag = SimpleRAG()

# Set username
rag.set_username("alice_smith")

# Test the application
print("Testing RAG application...")
test_response = rag.answer_query("What is machine learning?")
print(f"Response: {test_response}")

# Register with Snowflake AI Observability
connector = SnowflakeConnector(snowpark_session=session)
app_name = f"minimal_rag_username_test_{int(time.time())}"

tru_app = TruApp(
    rag,
    app_name=app_name,
    app_version="v1.0",
    connector=connector,
    main_method=rag.answer_query
)

print(f"Application registered: {app_name}")

# Create simple test dataset
test_data = pd.DataFrame({
    'query': [
        "What is AI?",
        "Tell me about machine learning",
        "What is Python?"
    ],
    'expected_answer': [
        "AI is artificial intelligence",
        "ML is machine learning",
        "Python is a programming language"
    ],
    'username': [
        "alice_smith",
        "bob_jones",
        "charlie_brown"
    ]
})

# Create run configuration
run_config = RunConfig(
    run_name=f"username_test_run_{int(time.time())}",
    description="Testing username tracking with custom span attributes",
    label="username_test",
    source_type="DATAFRAME",
    dataset_name="Username test dataset",
    dataset_spec={
        "RETRIEVAL.QUERY_TEXT": "query",
        "RECORD_ROOT.INPUT": "query",
        "RECORD_ROOT.GROUND_TRUTH_OUTPUT": "expected_answer",
    },
    llm_judge_name="mistral-large2"
)

# Add and execute run
run = tru_app.add_run(run_config=run_config)
print(f"Run created: {run_config.run_name}")

# Execute with different usernames
print("\nExecuting queries with different usernames...")
for index, row in test_data.iterrows():
    rag.set_username(row['username'])
    response = rag.answer_query(row['query'])
    print(f"User: {row['username']}, Query: {row['query']}, Response: {response[:50]}...")

# Wait for completion
print("\nWaiting for run completion...")
max_attempts = 30
for attempt in range(max_attempts):
    status = run.get_status()
    print(f"Status: {status}")
    if status in ["INVOCATION_COMPLETED", "INVOCATION_PARTIALLY_COMPLETED"]:
        break
    time.sleep(10)

# Compute supported metrics
print("\nComputing supported metrics...")
try:
    run.compute_metrics(metrics=["answer_relevance", "context_relevance", "correctness"])
    print("✅ Metrics computation initiated")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60)
print("HOW TO VIEW USERNAME AND CUSTOM DATA")
print("="*60)
print("1. In Snowsight:")
print("   - Go to AI & ML → Evaluations")
print("   - Select your application")
print("   - Click on individual records")
print("   - View trace details to see custom attributes")
print("   - Look for: user.name, custom.toxicity_score, custom.query_length")

print("\n2. Via SQL Query:")
print("""
SELECT 
    trace_id,
    span_name,
    span_attributes:"user.name" as username,
    span_attributes:"custom.toxicity_score" as toxicity_score,
    span_attributes:"custom.query_length" as query_length,
    span_attributes:"custom.topic_matched" as topic_matched,
    input_text,
    output_text
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS
WHERE span_attributes:"user.name" IS NOT NULL
ORDER BY start_time DESC
LIMIT 10;
""")

print("\n✅ KEY POINTS:")
print("- Username stored as: span_attributes:\"user.name\"")
print("- Custom metrics stored as: span_attributes:\"custom.*\"")
print("- GENERATION span has NO @instrument attributes (per documentation)")
print("- Custom data only via OpenTelemetry span.set_attribute()")
print("- This is NOT dashboard metrics - only trace-level data")
print("- Only 5 real metrics supported: answer_relevance, context_relevance, groundedness, correctness, coherence")
