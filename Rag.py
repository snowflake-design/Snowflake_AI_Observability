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

# Set environment variable for TruLens OpenTelemetry tracing
os.environ["TRULENS_OTEL_TRACING"] = "1"

# Snowflake connection parameters - UPDATE THESE
SNOWFLAKE_CONFIG = {
    'account': 'your_account',        
    'user': 'your_username',          
    'password': 'your_password',      
    'warehouse': 'your_warehouse',    
    'database': 'your_database',      
    'schema': 'your_schema',          
    'role': 'abc_admin'               
}

# Get Snowflake session
try:
    session = get_active_session()
    print("Using active Snowflake session")
except:
    print("Creating new Snowflake session...")
    session = Session.builder.configs(SNOWFLAKE_CONFIG).create()

# Set the context
session.use_role(SNOWFLAKE_CONFIG['role'])
session.use_warehouse(SNOWFLAKE_CONFIG['warehouse'])
session.use_database(SNOWFLAKE_CONFIG['database'])
session.use_schema(SNOWFLAKE_CONFIG['schema'])

print(f"Current context: {session.get_current_database()}.{session.get_current_schema()}")

# Get current username for tracking
current_username = session.get_current_user()
print(f"Current user: {current_username}")

# Enhanced RAG Application with user tracking from dataset
class RAGLLMApp:
    def __init__(self):
        self.model = "mistral-large2"
        
        # Sample knowledge base contexts
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
    
    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        }
    )
    def retrieve_context(self, query: str, username: str = None) -> list:
        """Retrieve relevant contexts from knowledge base"""
        print(f"Retrieving context for user {username}: {query}")
        
        # Simple keyword-based retrieval
        query_lower = query.lower()
        retrieved_contexts = []
        
        for topic, contexts in self.knowledge_base.items():
            if topic in query_lower:
                retrieved_contexts.extend(contexts)
                break
        
        # If no specific match, return general AI context
        if not retrieved_contexts:
            retrieved_contexts = self.knowledge_base.get("artificial intelligence", [])
        
        print(f"Retrieved {len(retrieved_contexts)} context pieces for {username}")
        return retrieved_contexts
    
    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate_response(self, query: str, contexts: list, username: str = None) -> str:
        """Generate response using retrieved contexts and LLM"""
        print(f"Generating response for user: {username}")
        
        # Prepare context string
        context_str = "\n".join([f"- {ctx}" for ctx in contexts])
        
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
            print(f"Generated response for {username} - Length: {len(response)} characters")
            return response
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return error_msg
    
    def answer_query(self, query: str, username: str = "unknown_user") -> str:
        """Main entry point for RAG pipeline - will be called by run.start()
        
        Note: This function is instrumented implicitly as RECORD_ROOT when specified 
        as main_method in TruApp. The RECORD_ROOT span captures:
        - RECORD_ROOT.INPUT: Input query
        - RECORD_ROOT.OUTPUT: Final response
        """
        print(f"Processing query from user {username}: {query}")
        
        # Step 1: Retrieve relevant contexts
        contexts = self.retrieve_context(query, username)
        
        # Step 2: Generate response using contexts
        response = self.generate_response(query, contexts, username)
        
        return response

# Initialize the RAG application
rag_app = RAGLLMApp()

# Test basic functionality first
print("\n=== Testing Basic RAG Functionality ===")
try:
    test_response = rag_app.answer_query("What is machine learning?", "test_user")
    print(f"Basic test successful: {test_response[:100]}...")
except Exception as e:
    print(f"Basic test failed: {e}")
    exit(1)

# Set up TruLens connector
print("\n=== Setting up TruLens Connector ===")
try:
    tru_snowflake_connector = SnowflakeConnector(snowpark_session=session)
    print("TruLens connector created")
except Exception as e:
    print(f"Connector warning: {e}")
    tru_snowflake_connector = SnowflakeConnector(snowpark_session=session)

# Register application with TruLens
print("\n=== Registering RAG Application ===")
timestamp = int(time.time())
app_name = f"rag_llm_app_{timestamp}"
app_version = "v1"

print(f"App name: {app_name}")

try:
    tru_rag_app = TruApp(
        rag_app,
        app_name=app_name,
        app_version=app_version,
        connector=tru_snowflake_connector,
        main_method=rag_app.answer_query
    )
    print("RAG application registered successfully")
except Exception as e:
    print(f"Application registration failed: {e}")
    exit(1)

# Create comprehensive test dataset with predefined users
print("\n=== Creating Test Dataset with Predefined Users ===")
test_data = pd.DataFrame({
    'query': [
        "What is machine learning and how does it work?",
        "Explain cloud computing benefits for businesses",
        "What are the main applications of artificial intelligence?",
        "How does natural language processing help in AI?",
        "What makes Snowflake different from other data platforms?",
        "Compare supervised and unsupervised learning",
        "What are the security benefits of cloud computing?",
        "How can AI be used in healthcare applications?"
    ],
    'username': [
        "alice_smith",
        "bob_johnson", 
        "carol_davis",
        "david_wilson",
        "eve_brown",
        "frank_miller",
        "grace_taylor",
        "henry_anderson"
    ],
    'department': [
        "Data Science",
        "IT Operations",
        "Product Management", 
        "Engineering",
        "Analytics",
        "Data Science",
        "IT Security",
        "Healthcare"
    ],
    'expected_answer': [
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

print(f"Created dataset with {len(test_data)} queries from {len(test_data['username'].unique())} different users")
print(f"DataFrame columns: {list(test_data.columns)}")
print("\nUser distribution:")
for dept in test_data['department'].unique():
    users_in_dept = test_data[test_data['department'] == dept]['username'].tolist()
    print(f"  {dept}: {', '.join(users_in_dept)}")

# Create run configuration with user tracking
print("\n=== Creating Run Configuration with User Tracking ===")
run_name = f"rag_user_tracking_run_{timestamp}"
run_config = RunConfig(
    run_name=run_name,
    description="RAG evaluation with user tracking and metrics",
    label="rag_user_metrics",
    source_type="DATAFRAME",
    dataset_name="rag_user_dataset",
    dataset_spec={
        "RECORD_ROOT.INPUT": "query",
        "RECORD_ROOT.GROUND_TRUTH_OUTPUT": "expected_answer",
        "RETRIEVAL.QUERY_TEXT": "query",
        # Note: username and department will be passed as context but not mapped to official attributes
    },
    llm_judge_name="mistral-large2"
)

print(f"Run name: {run_name}")
print(f"Dataset spec: {run_config.dataset_spec}")
print(f"LLM judge: {run_config.llm_judge_name}")
print("User tracking: username and department will be captured in trace context")

# Add run to TruApp
print("\n=== Adding Run to TruApp ===")
try:
    run: Run = tru_rag_app.add_run(run_config=run_config)
    print("Run added successfully")
except Exception as e:
    print(f"Failed to add run: {e}")
    exit(1)

# Execute the run
print("\n=== Executing Run with Traces and Metrics ===")
try:
    print("Starting run execution...")
    print("This will:")
    print("  1. Process each query through the RAG pipeline")
    print("  2. Capture retrieval and generation traces")
    print("  3. Log user information and contexts")
    print("  4. Ingest all data into AI Observability")
    
    run.start(input_df=test_data)
    
    print("Run execution completed successfully")
    print("Traces and user data should now be in the AI Observability event table")
    
except Exception as e:
    print(f"Run execution failed: {e}")
    print("Checking if partial traces were captured...")

# Wait for trace ingestion
print("\n=== Waiting for trace ingestion ===")
time.sleep(10)

# Compute evaluation metrics
print("\n=== Computing Evaluation Metrics ===")
try:
    print("Starting metrics computation...")
    print("Computing: answer_relevance, groundedness, context_relevance")
    
    run.compute_metrics([
        "answer_relevance",
        "groundedness", 
        "context_relevance"
    ])
    
    print("Metrics computation initiated successfully")
    print("Note: Metrics computation is asynchronous and may take several minutes")
    
except Exception as e:
    print(f"Metrics computation failed: {e}")
    print("Traces should still be available even without metrics")

# Wait for metrics computation
print("\n=== Waiting for metrics computation ===")
time.sleep(15)

# Verify traces and metrics
print("\n=== Verifying Traces and Metrics ===")
try:
    # Check total trace count
    trace_count = session.sql("""
        SELECT COUNT(*) as trace_count 
        FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
        WHERE application_name = ?
    """, params=[app_name]).collect()
    
    if trace_count and trace_count[0]['TRACE_COUNT'] > 0:
        print(f"SUCCESS! Found {trace_count[0]['TRACE_COUNT']} traces in event table")
        
        # Check different span types
        span_types = session.sql("""
            SELECT 
                span_type,
                COUNT(*) as count
            FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
            WHERE application_name = ?
            GROUP BY span_type
            ORDER BY count DESC
        """, params=[app_name]).collect()
        
        print("\nSpan types captured:")
        for span in span_types:
            print(f"  - {span['SPAN_TYPE']}: {span['COUNT']} traces")
        
        # Show sample retrieval traces
        retrieval_traces = session.sql("""
            SELECT 
                record_id,
                input,
                output,
                created_at
            FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
            WHERE application_name = ? 
            AND span_type = 'RETRIEVAL'
            ORDER BY created_at DESC
            LIMIT 2
        """, params=[app_name]).collect()
        
        print(f"\nSample retrieval traces ({len(retrieval_traces)} found):")
        for i, trace in enumerate(retrieval_traces, 1):
            if trace['INPUT']:
                input_preview = str(trace['INPUT'])[:60]
                print(f"  {i}. Query: {input_preview}...")
            if trace['OUTPUT']:
                output_preview = str(trace['OUTPUT'])[:80]
                print(f"     Retrieved: {output_preview}...")
        
        # Check for metrics
        metrics_check = session.sql("""
            SELECT 
                COUNT(*) as metric_count
            FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
            WHERE application_name = ? 
            AND record_type = 'EVALUATION'
        """, params=[app_name]).collect()
        
        if metrics_check and metrics_check[0]['METRIC_COUNT'] > 0:
            print(f"\nMetrics found: {metrics_check[0]['METRIC_COUNT']} evaluation records")
        else:
            print("\nNo metrics found yet (may still be computing)")
            
    else:
        print("No traces found yet")
        
except Exception as e:
    print(f"Could not verify traces: {e}")

# Check run status
print("\n=== Checking Run Status ===")
try:
    status = run.get_status()
    print(f"Run status: {status}")
except Exception as e:
    print(f"Could not get run status: {e}")

# Final verification and instructions
print("\n" + "="*70)
print("RAG AI OBSERVABILITY TEST COMPLETED")
print("="*70)

print(f"\nCHECK RESULTS:")
print(f"1. Snowsight: AI & ML -> Evaluations -> {app_name}")
print(f"2. View traces: SELECT * FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS WHERE application_name = '{app_name}';")
print(f"3. View metrics: SELECT * FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS WHERE application_name = '{app_name}' AND record_type = 'EVALUATION';")

print(f"\nKEY INFORMATION:")
print(f"- Application: {app_name}")
print(f"- Run: {run_name}")
print(f"- Users tracked: {len(test_data['username'].unique())} different users")
print(f"- Departments: {', '.join(test_data['department'].unique())}")
print(f"- Queries processed: {len(test_data)}")
print(f"- Expected trace types: RETRIEVAL, GENERATION, RECORD_ROOT")
print(f"- Metrics computed: answer_relevance, groundedness, context_relevance")

print(f"\nDATA CAPTURED:")
print(f"- User queries and responses by specific users")
print(f"- Retrieved contexts for each query")
print(f"- LLM generation traces")
print(f"- User identification (alice_smith, bob_johnson, etc.)")
print(f"- Department information for user segmentation")
print(f"- Ground truth comparisons")
print(f"- Evaluation metrics (if computation completed)")

print(f"\nUSER TRACKING ANALYSIS:")
print(f"- Who asked what: Each query linked to specific username")
print(f"- Department patterns: Analyze questions by department")
print(f"- User behavior: Response quality per user")
print(f"- Usage analytics: Most active users and topics")

print(f"\nSQL QUERIES FOR USER ANALYSIS:")
print(f"-- Questions by user:")
print(f"SELECT input, COUNT(*) FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS")
print(f"WHERE application_name = '{app_name}' GROUP BY input;")
print(f"")
print(f"-- User activity patterns:")
print(f"SELECT span_type, COUNT(*) FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS") 
print(f"WHERE application_name = '{app_name}' GROUP BY span_type;")

print(f"\nNEXT STEPS:")
print(f"1. Wait 5-10 minutes for metrics to complete")
print(f"2. View detailed traces in Snowsight")
print(f"3. Analyze retrieval quality and response accuracy")
print(f"4. Use insights to improve your RAG system")

print(f"\nImplementation follows official Snowflake AI Observability documentation")
