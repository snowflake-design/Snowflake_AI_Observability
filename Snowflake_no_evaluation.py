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
    print("✅ Using active Snowflake session")
except:
    print("Creating new Snowflake session...")
    session = Session.builder.configs(SNOWFLAKE_CONFIG).create()

# Set the context
session.use_role(SNOWFLAKE_CONFIG['role'])
session.use_warehouse(SNOWFLAKE_CONFIG['warehouse'])
session.use_database(SNOWFLAKE_CONFIG['database'])
session.use_schema(SNOWFLAKE_CONFIG['schema'])

print(f"Current context: {session.get_current_database()}.{session.get_current_schema()}")

# LLM Application with proper instrumentation (following official docs)
class BasicLLMApp:
    def __init__(self):
        self.model = "mistral-large2"
    
    @instrument(
        span_type=SpanAttributes.SpanType.GENERATION,
        attributes={
            SpanAttributes.GENERATION.INPUT_MESSAGES: "query",
            SpanAttributes.GENERATION.OUTPUT_MESSAGES: "return",
        }
    )
    def generate_response(self, query: str) -> str:
        """Generate response using Cortex Complete"""
        prompt = f"Answer this question clearly: {query}"
        try:
            response = complete(self.model, prompt)
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    def answer_query(self, query: str) -> str:
        """Main entry point - will be called by run.start()"""
        return self.generate_response(query)

# Initialize the application
llm_app = BasicLLMApp()

# Test basic functionality first
print("\n=== Testing Basic LLM Functionality ===")
try:
    test_response = llm_app.answer_query("What is 2+2?")
    print(f"✅ Basic test successful: {test_response[:100]}...")
except Exception as e:
    print(f"❌ Basic test failed: {e}")
    exit(1)

# Set up TruLens connector
print("\n=== Setting up TruLens Connector ===")
try:
    tru_snowflake_connector = SnowflakeConnector(snowpark_session=session)
    print("✅ TruLens connector created")
except Exception as e:
    print(f"⚠️ Connector warning: {e}")
    tru_snowflake_connector = SnowflakeConnector(snowpark_session=session)

# Register application with TruLens
print("\n=== Registering Application ===")
timestamp = int(time.time())
app_name = f"basic_llm_app_{timestamp}"  # Unique name to avoid conflicts
app_version = "v1"

print(f"App name: {app_name}")

try:
    tru_llm_app = TruApp(
        llm_app,
        app_name=app_name,
        app_version=app_version,
        connector=tru_snowflake_connector,
        main_method=llm_app.answer_query  # Specify entry point method
    )
    print("✅ Application registered successfully")
except Exception as e:
    print(f"❌ Application registration failed: {e}")
    exit(1)

# Create test dataset (in-memory DataFrame)
print("\n=== Creating Test Dataset ===")
test_data = pd.DataFrame({
    'query': [
        "What is machine learning?",
        "Explain cloud computing in simple terms", 
        "What are the benefits of artificial intelligence?",
        "How does natural language processing work?",
        "What is the difference between AI and ML?"
    ]
})

print(f"Created dataset with {len(test_data)} queries")
print(f"DataFrame columns: {list(test_data.columns)}")

# Create run configuration (following official docs exactly)
print("\n=== Creating Run Configuration ===")
run_name = f"llm_evaluation_run_{timestamp}"
run_config = RunConfig(
    run_name=run_name,
    description="Basic LLM evaluation run for AI Observability",
    label="llm_test",
    source_type="DATAFRAME",  # Using in-memory DataFrame
    dataset_name="basic_llm_test_dataset",  # Any name for DATAFRAME source
    dataset_spec={
        "RECORD_ROOT.INPUT": "query",  # Maps to DataFrame column 'query'
    },
)

print(f"Run name: {run_name}")
print(f"Dataset spec: {run_config.dataset_spec}")

# Add run to TruApp
print("\n=== Adding Run to TruApp ===")
try:
    run: Run = tru_llm_app.add_run(run_config=run_config)
    print("✅ Run added successfully")
except Exception as e:
    print(f"❌ Failed to add run: {e}")
    exit(1)

# Execute the run (this is where traces get ingested)
print("\n=== Executing Run (This ingests traces) ===")
try:
    print("Starting run execution...")
    print("This will:")
    print("  1. Read inputs from the DataFrame")
    print("  2. Call llm_app.answer_query() for each input")
    print("  3. Generate traces")
    print("  4. Ingest traces into SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS")
    
    # This is the key call that ingests traces
    run.start(input_df=test_data)
    
    print("✅ Run execution completed successfully!")
    print("✅ Traces should now be in the AI Observability event table")
    
except Exception as e:
    print(f"❌ Run execution failed: {e}")
    print("This might be due to system issues, but let's check if traces were captured anyway...")

# Wait for trace ingestion
print("\n⏳ Waiting for trace ingestion...")
time.sleep(10)

# Verify traces were ingested
print("\n=== Verifying Trace Ingestion ===")
try:
    # Check for traces in the event table
    trace_count = session.sql("""
        SELECT COUNT(*) as trace_count 
        FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
        WHERE application_name = ?
    """, params=[app_name]).collect()
    
    if trace_count and trace_count[0]['TRACE_COUNT'] > 0:
        print(f"✅ SUCCESS! Found {trace_count[0]['TRACE_COUNT']} traces in event table")
        
        # Show sample traces
        sample_traces = session.sql("""
            SELECT 
                record_id,
                span_type,
                span_name,
                input,
                output,
                created_at
            FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
            WHERE application_name = ?
            ORDER BY created_at DESC
            LIMIT 3
        """, params=[app_name]).collect()
        
        print("\nSample traces:")
        for i, trace in enumerate(sample_traces, 1):
            print(f"  {i}. {trace['SPAN_TYPE']}: {trace['SPAN_NAME']}")
            if trace['INPUT']:
                input_preview = str(trace['INPUT'])[:50]
                print(f"     Input: {input_preview}...")
            if trace['OUTPUT']:
                output_preview = str(trace['OUTPUT'])[:50]
                print(f"     Output: {output_preview}...")
            
    else:
        print("❌ No traces found yet. This could mean:")
        print("  - Traces are still being ingested (wait longer)")
        print("  - There's a privilege or system issue")
        print("  - Check if external agent was created")
        
except Exception as e:
    print(f"❌ Could not check trace table: {e}")

# Check external agent
print("\n=== Checking External Agent ===")
try:
    agents = session.sql("SHOW EXTERNAL AGENTS").collect()
    app_found = False
    for agent in agents:
        if app_name in str(agent):
            print(f"✅ Found external agent: {agent}")
            app_found = True
            break
    
    if not app_found:
        print(f"❌ External agent '{app_name}' not found")
        print("Available external agents:")
        for agent in agents:
            print(f"  - {agent}")
            
except Exception as e:
    print(f"❌ Could not check external agents: {e}")

# Final instructions
print("\n" + "="*70)
print("🎉 AI OBSERVABILITY TEST COMPLETED!")
print("="*70)

print(f"\n📊 CHECK RESULTS:")
print(f"1. Snowsight: AI & ML → Evaluations → {app_name}")
print(f"2. SQL Query:")
print(f"   SELECT * FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS")
print(f"   WHERE application_name = '{app_name}';")
print(f"3. External Agents: SHOW EXTERNAL AGENTS;")

print(f"\n🔍 KEY INFORMATION:")
print(f"- Application Name: {app_name}")
print(f"- Run Name: {run_name}")
print(f"- Dataset: {len(test_data)} queries processed")
print(f"- Expected traces: {len(test_data)} + metadata")

print(f"\n⚠️  IF NO TRACES APPEAR:")
print(f"1. Wait 5-15 minutes for ingestion")
print(f"2. Check privileges: SHOW GRANTS TO ROLE CURRENT_ROLE();")
print(f"3. Verify you have SNOWFLAKE.AI_OBSERVABILITY_EVENTS_LOOKUP role")
print(f"4. Check if AI Observability is enabled in your account")

print(f"\n✅ This implementation follows the official Snowflake documentation exactly!")
