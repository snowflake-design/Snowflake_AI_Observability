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

# Enhanced RAG Application with FIXED instrumentation
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
    def retrieve_context(self, query: str) -> list:
        """Retrieve relevant contexts from knowledge base"""
        print(f"Retrieving context for query: {query}")
        
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
        
        print(f"Retrieved {len(retrieved_contexts)} context pieces")
        return retrieved_contexts
    
    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate_response(self, query: str, contexts: list) -> str:
        """Generate response using retrieved contexts and LLM"""
        print(f"Generating response for query: {query}")
        
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
            print(f"Generated response - Length: {len(response)} characters")
            return response
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return error_msg
    
    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        })
    def query(self, query: str) -> str:
        """Main entry point for RAG pipeline - FIXED method name and instrumentation"""
        print(f"Processing query: {query}")
        
        # Step 1: Retrieve relevant contexts
        contexts = self.retrieve_context(query)
        
        # Step 2: Generate response using contexts
        response = self.generate_response(query, contexts)
        
        return response

# Initialize the RAG application
rag_app = RAGLLMApp()

# Test basic functionality first
print("\n=== Testing Basic RAG Functionality ===")
try:
    test_response = rag_app.query("What is machine learning?")
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
    # FIXED: Use the correct method name and no main_method specification
    tru_rag_app = TruApp(
        rag_app,
        app_name=app_name,
        app_version=app_version,
        connector=tru_snowflake_connector
    )
    print("RAG application registered successfully")
except Exception as e:
    print(f"Application registration failed: {e}")
    exit(1)

# Create comprehensive test dataset with CORRECT column mapping
print("\n=== Creating Test Dataset with Correct Column Names ===")
test_data = pd.DataFrame({
    'query': [  # This will map to RECORD_ROOT.INPUT
        "What is machine learning and how does it work?",
        "Explain cloud computing benefits for businesses",
        "What are the main applications of artificial intelligence?",
        "How does natural language processing help in AI?",
        "What makes Snowflake different from other data platforms?",
        "Compare supervised and unsupervised learning",
        "What are the security benefits of cloud computing?",
        "How can AI be used in healthcare applications?"
    ],
    'expected_answer': [  # This will map to RECORD_ROOT.GROUND_TRUTH_OUTPUT
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

print(f"Created dataset with {len(test_data)} queries")
print(f"DataFrame columns: {list(test_data.columns)}")

# FIXED: Create run configuration with CORRECT dataset specification
print("\n=== Creating Run Configuration with FIXED Dataset Spec ===")
run_name = f"rag_metrics_test_run_{timestamp}"
run_config = RunConfig(
    run_name=run_name,
    description="RAG evaluation with FIXED metrics computation",
    label="rag_metrics_test",
    source_type="DATAFRAME",
    dataset_name="rag_test_dataset",
    dataset_spec={
        "input": "query",  # Maps to DataFrame column name
        "ground_truth_output": "expected_answer",  # Maps to DataFrame column name
    },
    llm_judge_name="mistral-large2"
)

print(f"Run name: {run_name}")
print(f"FIXED Dataset spec: {run_config.dataset_spec}")
print(f"LLM judge: {run_config.llm_judge_name}")

# Add run to TruApp
print("\n=== Adding Run to TruApp ===")
try:
    run: Run = tru_rag_app.add_run(run_config=run_config)
    print("Run added successfully")
except Exception as e:
    print(f"Failed to add run: {e}")
    exit(1)

# Execute the run
print("\n=== Executing Run with FIXED Implementation ===")
try:
    print("Starting run execution...")
    print("This will:")
    print("  1. Process each query through the RAG pipeline")
    print("  2. Capture CORRECTLY instrumented traces")
    print("  3. Store data with PROPER attribute mapping")
    print("  4. Enable metrics computation")
    
    run.start(input_df=test_data)
    
    print("Run execution completed successfully")
    print("Waiting for trace ingestion before computing metrics...")
    
except Exception as e:
    print(f"Run execution failed: {e}")
    print("Checking if partial traces were captured...")

# CRITICAL: Wait for trace ingestion before computing metrics
print("\n=== Waiting for Trace Ingestion (IMPORTANT) ===")
time.sleep(30)  # Increased wait time

# Verify traces exist before computing metrics
print("\n=== Verifying Traces Before Metrics Computation ===")
try:
    trace_count = session.sql("""
        SELECT COUNT(*) as trace_count 
        FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
        WHERE application_name = ?
    """, params=[app_name]).collect()
    
    if trace_count and trace_count[0]['TRACE_COUNT'] > 0:
        print(f"SUCCESS! Found {trace_count[0]['TRACE_COUNT']} traces")
        
        # Check span types
        span_types = session.sql("""
            SELECT 
                span_type,
                COUNT(*) as count
            FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
            WHERE application_name = ?
            GROUP BY span_type
            ORDER BY count DESC
        """, params=[app_name]).collect()
        
        print("Span types captured:")
        for span in span_types:
            print(f"  - {span['SPAN_TYPE']}: {span['COUNT']} traces")
            
        # Verify we have the required span types for metrics
        span_type_names = [span['SPAN_TYPE'] for span in span_types]
        required_spans = ['RECORD_ROOT', 'RETRIEVAL', 'GENERATION']
        
        missing_spans = [span for span in required_spans if span not in span_type_names]
        if missing_spans:
            print(f"WARNING: Missing required span types for metrics: {missing_spans}")
            print("Metrics computation may fail!")
        else:
            print("All required span types present for metrics computation")
            
    else:
        print("ERROR: No traces found - cannot compute metrics")
        exit(1)
        
except Exception as e:
    print(f"Could not verify traces: {e}")

# Compute evaluation metrics with PROPER timing
print("\n=== Computing Evaluation Metrics (FIXED) ===")
try:
    print("Starting metrics computation...")
    print("Computing: answer_relevance, groundedness, context_relevance")
    
    # These are the core RAG Triad metrics
    run.compute_metrics([
        "answer_relevance",   # Requires RECORD_ROOT.INPUT and RECORD_ROOT.OUTPUT
        "groundedness",       # Requires RETRIEVAL.RETRIEVED_CONTEXTS and RECORD_ROOT.OUTPUT  
        "context_relevance"   # Requires RETRIEVAL.QUERY_TEXT and RETRIEVAL.RETRIEVED_CONTEXTS
    ])
    
    print("Metrics computation initiated successfully")
    print("Note: Metrics computation is asynchronous and may take several minutes")
    
except Exception as e:
    print(f"Metrics computation failed: {e}")
    print("This could be due to:")
    print("1. Missing required attributes in spans")
    print("2. Incorrect dataset specification")
    print("3. Insufficient traces")
    print("4. LLM judge access issues")

# Wait for metrics computation
print("\n=== Waiting for Metrics Computation ===")
time.sleep(30)  # Wait for metrics to be computed

# Verify metrics were computed
print("\n=== Verifying Metrics Computation ===")
try:
    # Check for evaluation records
    metrics_check = session.sql("""
        SELECT 
            COUNT(*) as metric_count,
            COUNT(DISTINCT record_id) as unique_records
        FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
        WHERE application_name = ? 
        AND record_type = 'EVALUATION'
    """, params=[app_name]).collect()
    
    if metrics_check and metrics_check[0]['METRIC_COUNT'] > 0:
        print(f"SUCCESS! Found {metrics_check[0]['METRIC_COUNT']} evaluation records")
        print(f"Covering {metrics_check[0]['UNIQUE_RECORDS']} unique query records")
        
        # Show sample metrics
        sample_metrics = session.sql("""
            SELECT 
                metric_name,
                metric_value,
                record_id
            FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
            WHERE application_name = ? 
            AND record_type = 'EVALUATION'
            ORDER BY created_at DESC
            LIMIT 10
        """, params=[app_name]).collect()
        
        print("\nSample metrics computed:")
        for metric in sample_metrics:
            print(f"  - {metric['METRIC_NAME']}: {metric['METRIC_VALUE']:.3f}")
            
    else:
        print("No metrics found yet (may still be computing)")
        print("Check again in a few minutes, or there may be an issue with:")
        print("1. Span attribute mapping")
        print("2. Dataset specification")
        print("3. LLM judge permissions")
        
except Exception as e:
    print(f"Could not verify metrics: {e}")

# Final verification and instructions
print("\n" + "="*70)
print("FIXED RAG AI OBSERVABILITY TEST COMPLETED")
print("="*70)

print(f"\nCHECK RESULTS:")
print(f"1. Snowsight: AI & ML -> Evaluations -> {app_name}")
print(f"2. View traces: SELECT * FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS WHERE application_name = '{app_name}';")
print(f"3. View metrics: SELECT * FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS WHERE application_name = '{app_name}' AND record_type = 'EVALUATION';")
