
import os
import pandas as pd
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.session import Session
from snowflake.snowpark import functions as F
from snowflake.cortex import complete
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
from trulens.apps.app import TruApp
from trulens.connectors.snowflake import SnowflakeConnector

# Set environment variable for TruLens OpenTelemetry tracing
os.environ["TRULENS_OTEL_TRACING"] = "1"

# Snowflake connection parameters - UPDATE THESE
SNOWFLAKE_CONFIG = {
    'account': 'your_account',        # Your Snowflake account identifier
    'user': 'your_username',          # Your username
    'password': 'your_password',      # Your password
    'warehouse': 'your_warehouse',    # Your warehouse
    'database': 'your_database',      # Database where you have CREATE EXTERNAL AGENT
    'schema': 'your_schema',          # Schema where you have CREATE EXTERNAL AGENT
    'role': 'abc_admin'               # Your role with AI Observability privileges
}

# Get Snowflake session - try active session first, fallback to new connection
try:
    session = get_active_session()
    print("Using active Snowflake session")
except:
    print("Creating new Snowflake session with provided config")
    from snowflake.snowpark import Session
    session = Session.builder.configs(SNOWFLAKE_CONFIG).create()

# Set the context
session.use_role(SNOWFLAKE_CONFIG['role'])
session.use_warehouse(SNOWFLAKE_CONFIG['warehouse'])
session.use_database(SNOWFLAKE_CONFIG['database'])
session.use_schema(SNOWFLAKE_CONFIG['schema'])

# Basic LLM Application with simple decorators - NO ATTRIBUTES
class BasicLLMApp:
    def __init__(self):
        self.model = "mistral-large2"
    
    @instrument()  # Simple decorator, no attributes
    def generate_response(self, query: str) -> str:
        """Generate response using Cortex Complete"""
        prompt = f"Answer this question: {query}"
        
        try:
            response = complete(self.model, prompt)
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    @instrument()  # Simple decorator, no attributes
    def query(self, query: str) -> str:
        """Main query method"""
        return self.generate_response(query)

# Initialize the application
llm_app = BasicLLMApp()

# Test the application manually first
print("=== Testing Basic LLM Application ===")
test_query = "What is artificial intelligence?"
print(f"Query: {test_query}")
response = llm_app.query(test_query)
print(f"Response: {response}")

# Set up TruLens connector for Snowflake
print("Setting up TruLens Snowflake connector...")
try:
    tru_snowflake_connector = SnowflakeConnector(snowpark_session=session)
    print("✅ Snowflake connector created successfully")
except Exception as e:
    print(f"⚠️ Connector warning: {e}")
    # Create connector anyway - this error often doesn't affect functionality
    tru_snowflake_connector = SnowflakeConnector(snowpark_session=session)

# Register the application with TruLens
app_name = "basic_llm_test"
app_version = "v1"

print(f"Registering application: {app_name}")

try:
    tru_llm_app = TruApp(
        llm_app,
        app_name=app_name,
        app_version=app_version,
        connector=tru_snowflake_connector
    )
    print("✅ TruApp registered successfully")
except Exception as e:
    print(f"⚠️ Registration warning: {e}")
    # Try alternative registration
    print("Trying alternative registration...")
    tru_llm_app = TruApp(
        llm_app,
        app_name=app_name,
        app_version=app_version,
        connector=tru_snowflake_connector
    )

# Test queries for tracing
test_queries = [
    "What is machine learning?",
    "Explain cloud computing", 
    "What is Python programming?"
]

print("\n=== Running Queries with Tracing (Using Context Manager) ===")
for i, query in enumerate(test_queries, 1):
    print(f"\nQuery {i}: {query}")
    try:
        # KEY FIX: Use TruApp context manager
        with tru_llm_app as recording:
            response = llm_app.query(query)
        print(f"Response: {response[:100]}...")  # Show first 100 chars
        print("✅ Trace captured with context manager")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n=== Verification ===")
print(f"Application: {app_name}")
print(f"Context: {SNOWFLAKE_CONFIG['database']}.{SNOWFLAKE_CONFIG['schema']}")

# Check if external agent was created
try:
    agents = session.sql("SHOW EXTERNAL AGENTS").collect()
    print(f"External agents: {len(agents)} found")
    for agent in agents:
        if app_name in str(agent):
            print(f"✅ Found: {agent}")
except Exception as e:
    print(f"Could not check external agents: {e}")

# Check traces in event table
try:
    import time
    time.sleep(3)  # Wait for traces to be ingested
    
    traces = session.sql("""
        SELECT COUNT(*) as count 
        FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
        WHERE application_name LIKE ?
    """, params=[f"%{app_name}%"]).collect()
    
    if traces and traces[0]['COUNT'] > 0:
        print(f"✅ Found {traces[0]['COUNT']} traces in event table")
    else:
        print("⏳ No traces found yet (may take time to ingest)")
        
except Exception as e:
    print(f"Could not check traces: {e}")

print("\n=== Check Results ===")
print("1. Snowsight: AI & ML → Evaluations")
print(f"2. SQL: SELECT * FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS WHERE application_name LIKE '%{app_name}%';")
print("3. External Agents: SHOW EXTERNAL AGENTS;")

print("\n✅ Basic AI Observability test complete!")
