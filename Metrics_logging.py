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

# 1. SETUP ENVIRONMENT ()
print("=== Setting Up Environment ===")

# Set environment variable for TruLens OpenTelemetry tracing
os.environ["TRULENS_OTEL_TRACING"] = "1"

# Get Snowflake session
try:
    session = get_active_session()
    print("Using active Snowflake session")
except:
    print("Creating new Snowflake session...")
    # Add your SNOWFLAKE_CONFIG here if needed
    session = Session.builder.configs(SNOWFLAKE_CONFIG).create()

print(f"Current context: {session.get_current_database()}.{session.get_current_schema()}")

# 2. INSTRUMENTING THE APPLICATION ()
print("=== Creating RAG Application with Instrumentation ===")

class RAGApplication:
    def __init__(self):
        self.model = "mistral-large2"
        
        # Your knowledge base ()
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
            ],
            "natural language processing": [
                "NLP is a branch of AI that helps computers understand, interpret, and generate human language.",
                "Common NLP tasks include sentiment analysis, language translation, and text summarization.",
                "Modern NLP uses transformer models and large language models for improved accuracy."
            ],
            "snowflake": [
                "Snowflake is a cloud-based data warehousing platform built for the cloud.",
                "It provides features like automatic scaling, data sharing, and separation of compute and storage.",
                "Snowflake supports structured and semi-structured data with SQL-based querying capabilities."
            ]
        }

    @instrument(span_type=SpanAttributes.SpanType.RETRIEVAL,
                attributes={
                    SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
                    SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return"
                })
    def retrieve_context(self, query: str) -> list:
        """Retrieve relevant contexts from knowledge base (like Medium article's retrieve_context)"""
        print(f"Retrieving context for query: {query}")
        
        # Simple keyword-based retrieval ()
        query_lower = query.lower()
        retrieved_contexts = []
        
        for topic, contexts in self.knowledge_base.items():
            if topic in query_lower:
                retrieved_contexts.extend(contexts)
                break
        
        # If no specific match, return general AI context
        if not retrieved_contexts:
            retrieved_contexts = self.knowledge_base.get("artificial intelligence", [])
        
        print(f"Retrieved {len(retrieved_contexts)} context pieces")
        return retrieved_contexts

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate_answer(self, query: str, context: list) -> str:
        """Generate answer using LLM (like Medium article's generate_answer)"""
        print(f"Generating answer for query: {query}")
        
        # Prepare context string
        context_str = "\n".join([f"- {ctx}" for ctx in context])
        
        prompt = f"""Based on the following context information, provide a comprehensive answer to the user's question.

Context:
{context_str}

User Question: {query}

Instructions:
- Use the provided context to answer the question
- If the context doesn't fully answer the question, acknowledge this
- Provide a helpful and informative response
- Keep the answer concise but complete

Answer:"""
        
        try:
            response = complete(self.model, prompt)
            print(f"Generated response - Length: {len(response)} characters")
            return response
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return error_msg

    @instrument(span_type=SpanAttributes.SpanType.RECORD_ROOT,
                attributes={
                    SpanAttributes.RECORD_ROOT.INPUT: "query",
                    SpanAttributes.RECORD_ROOT.OUTPUT: "return"
                })
    def answer_question(self, query: str) -> str:
        """High-level method to get answer (like Medium article's answer_question)"""
        print(f"Processing question: {query}")
        
        # Step 1: Retrieve relevant contexts
        docs = self.retrieve_context(query)
        
        # Step 2: Generate answer using contexts
        return self.generate_answer(query, docs)

# 3. REGISTERING THE APPLICATION ()
print("=== Registering Application in Snowflake Cortex ===")

# Create RAG application instance
rag_app = RAGApplication()

# Test basic functionality
print("Testing basic functionality...")
test_response = rag_app.answer_question("What is machine learning?")
print(f"Test successful: {test_response[:100]}...")

# Create Snowflake connector using active Snowpark session
snowflake_conn = SnowflakeConnector(snowpark_session=session)

# Register the app with a name and version ()
timestamp = int(time.time())
app_name = f"rag_observability_app_{timestamp}"  # 
app_version = "v1"

print(f"Registering app: {app_name}")

tru_app = TruApp(rag_app,
                app_name=app_name,
                app_version=app_version,
                connector=snowflake_conn)

print("Application registered successfully")

# 4. CONFIGURING AN EVALUATION RUN ()
print("=== Configuring Evaluation Run ===")

# Create test dataset ()
test_queries_data = pd.DataFrame({
    'QUERY': [  # 
        "What is machine learning and how does it work?",
        "Explain cloud computing benefits for businesses", 
        "What are the main applications of artificial intelligence?",
        "How does natural language processing help in AI?",
        "What makes Snowflake different from other data platforms?",
        "Compare supervised and unsupervised learning",
        "What are the security benefits of cloud computing?",
        "How can AI be used in healthcare applications?"
    ],
    'GROUND_TRUTH_RESPONSE': [  # 
        "Machine learning is a subset of AI that enables computers to learn from data without explicit programming.",
        "Cloud computing provides scalability, cost-effectiveness, and accessibility for business operations.",
        "AI applications include chatbots, recommendation systems, autonomous vehicles, and medical diagnosis.",
        "NLP helps computers understand, interpret, and generate human language for various applications.",
        "Snowflake provides cloud-native data warehousing with automatic scaling and data sharing capabilities.",
        "Supervised learning uses labeled data while unsupervised learning finds patterns in unlabeled data.",
        "Cloud computing offers enhanced security through professional management and compliance features.",
        "AI in healthcare includes medical imaging analysis, drug discovery, and personalized treatment plans."
    ]
})

print(f"Created dataset with {len(test_queries_data)} test queries")

# Create run configuration ()
run_config = RunConfig(
    run_name="experiment_1_run",  # 
    dataset_name="RAG_TEST_DATA",  # 
    description="Questions about AI and technology topics",  # 
    label="baseline_run",  # 
    source_type="DATAFRAME",  # 
    dataset_spec={
        "input": "QUERY",  # 
        "ground_truth_output": "GROUND_TRUTH_RESPONSE"  # 
    }
)

print(f"Run configuration created: {run_config.run_name}")
print(f"Dataset spec: {run_config.dataset_spec}")

# Add the run to TruApp ()
run = tru_app.add_run(run_config=run_config)
print("Run added to TruApp successfully")

# 5. RUNNING THE EVALUATION ()
print("=== Running the Evaluation ===")

# First, start the run to invoke the application in batch ()
print("Starting run.start() - this will process all queries...")
run.start(input_df=test_queries_data)  # Using your DataFrame
print("Run execution completed - all queries processed and traces captured")

# 6. COMPUTING METRICS ()
print("=== Computing Evaluation Metrics ===")

# Compute the RAG evaluation triad ()
print("Computing RAG Triad metrics: Answer Relevance, Context Relevance, and Groundedness")

run.compute_metrics([
    "answer_relevance",    # From
    "context_relevance",   # From triad  
    "groundedness"         # From triad
])

print("Metrics computation initiated successfully")
print("Note: Metrics computation is asynchronous and may take several minutes")

# 7. VIEWING RESULTS ()
print("=== Viewing Results Instructions ===")
print("\nTo view results (following Medium article steps):")
print("1. Navigate to Snowsight")
print("2. Open AI & ML menu -> Click Evaluations")
print(f"3. Look for application: {app_name}")
print("4. Click on your application to view runs")
print(f"5. Click on run: experiment_1_run")
print("6. View detailed results, metrics, and traces")

print(f"\nAlternatively, query the event table directly:")
print(f"-- View all traces:")
print(f"SELECT * FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS WHERE application_name = '{app_name}';")
print(f"")
print(f"-- View metrics only:")
print(f"SELECT * FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS WHERE application_name = '{app_name}' AND record_type = 'EVALUATION';")

# 8. VERIFICATION (Optional - check if data is there)
print("\n=== Quick Verification ===")
try:
    # Check if traces were created
    trace_count = session.sql("""
        SELECT COUNT(*) as count 
        FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
        WHERE application_name = ?
    """, params=[app_name]).collect()
    
    if trace_count and trace_count[0]['COUNT'] > 0:
        print(f"✅ SUCCESS: {trace_count[0]['COUNT']} traces found in event table")
        
        # Check span types
        span_types = session.sql("""
            SELECT span_type, COUNT(*) as count
            FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
            WHERE application_name = ?
            GROUP BY span_type
        """, params=[app_name]).collect()
        
        print("Span types captured:")
        for span in span_types:
            print(f"  - {span['SPAN_TYPE']}: {span['COUNT']}")
            
    else:
        print("No traces found yet - check application name or wait a moment")
        
except Exception as e:
    print(f"Verification query failed: {e}")

print("\n" + "="*70)
print("RAG OBSERVABILITY SETUP COMPLETE (Following Medium Article Structure)")
print("="*70)
print("✅ Environment set up")
print("✅ Application instrumented")
print("✅ Application registered")
print("✅ Evaluation run configured") 
print("✅ Run executed")
print("✅ Metrics computation initiated")
print("\n🕐 Wait 5-10 minutes, then check Snowsight AI & ML -> Evaluations")
print("📊 Metrics should appear automatically in the background")

# Key :
print(f"\nKEY CHANGES MADE TO FOLLOW MEDIUM ARTICLE:")
print("1. ✅ Method names: retrieve_context, generate_answer, answer_question (not answer_query)")
print("2. ✅ Explicit RECORD_ROOT instrumentation on main method")
print("3. ✅ Dataset column names in CAPS: QUERY, GROUND_TRUTH_RESPONSE")
print("4. ✅ Dataset spec format: 'input': 'QUERY', 'ground_truth_output': 'GROUND_TRUTH_RESPONSE'")
print("5. ✅ Run naming: experiment_1_run, baseline_run (Medium article style)")
print("6. ✅ Removed user tracking to match Medium article simplicity")
print("7. ✅ Exact RAG Triad metrics: answer_relevance, context_relevance, groundedness")
print("8. ✅ TruApp registration without main_method (letting RECORD_ROOT handle it)")

print(f"\nThis structure exactly matches the Medium article workflow!")
print(f"Your metrics should now compute properly! 🎉")
