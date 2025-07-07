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
from trulens.core.run import Run, RunConfig

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

# Simple LLM Application with Cortex Complete
class SimpleLLMApp:
    def __init__(self):
        self.model = "mistral-large2"  # or use "llama3.1-70b", "claude-3.5-sonnet", etc.
    
    @instrument(
        span_type=SpanAttributes.SpanType.GENERATION,
        attributes={
            SpanAttributes.GENERATION.INPUT_MESSAGES: "query",
            SpanAttributes.GENERATION.OUTPUT_MESSAGES: "return",
        }
    )
    def generate_response(self, query: str) -> str:
        """Generate response using Cortex Complete"""
        prompt = f"""
        You are a helpful assistant. Answer the following question clearly and concisely.
        
        Question: {query}
        
        Answer:
        """
        
        try:
            # Using Cortex Complete function
            response = complete(self.model, prompt)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        }
    )
    def query(self, query: str) -> str:
        """Main query method - this is the entry point for evaluation"""
        return self.generate_response(query)

# Initialize the application
llm_app = SimpleLLMApp()

# Test the application manually first
print("Testing the application manually:")
test_response = llm_app.query("What is artificial intelligence?")
print(f"Response: {test_response}")

# Set up TruLens connector for Snowflake
tru_snowflake_connector = SnowflakeConnector(snowpark_session=session)

# Register the application with TruLens
# Option 1: Simple name (TruLens will create in current schema)
app_name = "simple_llm_test"
app_version = "v1"

# Option 2: Fully qualified name (explicit control over location)
# app_name = f"{SNOWFLAKE_CONFIG['database']}.{SNOWFLAKE_CONFIG['schema']}.simple_llm_test"

print(f"Registering application: {app_name}")
print(f"Current context: {session.get_current_database()}.{session.get_current_schema()}")

tru_llm_app = TruApp(
    llm_app,
    app_name=app_name,
    app_version=app_version,
    connector=tru_snowflake_connector
)

# Verify external agent creation
try:
    print(f"External agent will be created as: {app_name}")
    # You can also check if it exists by trying to describe it
    agent_check = session.sql(f"DESC EXTERNAL AGENT {app_name}").collect()
    print(f"External agent {app_name} already exists")
except Exception as e:
    print(f"External agent {app_name} will be created during run registration")

# Create a simple test dataset
test_data = pd.DataFrame({
    'QUERY': [
        "What is artificial intelligence?",
        "Explain machine learning in simple terms",
        "What are the benefits of cloud computing?",
        "How does natural language processing work?",
        "What is the difference between AI and ML?"
    ],
    'EXPECTED_ANSWER': [
        "AI is computer technology that can perform tasks typically requiring human intelligence",
        "ML is a method of teaching computers to learn patterns from data",
        "Cloud computing offers scalability, cost-effectiveness, and accessibility",
        "NLP enables computers to understand and process human language",
        "AI is broader field, ML is a subset of AI focused on learning from data"
    ]
})

# Upload test dataset to Snowflake
print("Uploading test dataset to Snowflake...")
session.write_pandas(test_data, "TEST_QUERIES", overwrite=True)
print("Test dataset uploaded successfully!")

# Configure the evaluation run
run_name = "simple_llm_evaluation_run"

# Option 1: Simple dataset name (uses current schema context)
dataset_name = "TEST_QUERIES"

# Option 2: Fully qualified dataset name (explicit schema reference)
# dataset_name = f"{SNOWFLAKE_CONFIG['database']}.{SNOWFLAKE_CONFIG['schema']}.TEST_QUERIES"

print(f"Using dataset: {dataset_name}")

run_config = RunConfig(
    run_name=run_name,
    dataset_name=dataset_name,
    description="Simple LLM evaluation test with flexible naming",
    label="llm_test",
    source_type="TABLE",
    dataset_spec={
        "input": "QUERY",
        "ground_truth_output": "EXPECTED_ANSWER",
    },
)

# Add the run to TruLens
print(f"Adding run: {run_name} for app: {app_name}")
run: Run = tru_llm_app.add_run(run_config=run_config)

print(f"Starting evaluation run: {run_name}")
print("This will create the external agent if it doesn't exist...")

# Start the evaluation run
try:
    run.start()
    print("✅ Run completed successfully!")
except Exception as e:
    print(f"❌ Error during run: {e}")
    # If error mentions external agent not found, try with fully qualified name
    if "external agent" in str(e).lower() and "not exist" in str(e).lower():
        print("\n🔧 Trying with fully qualified external agent name...")
        qualified_app_name = f"{SNOWFLAKE_CONFIG['database']}.{SNOWFLAKE_CONFIG['schema']}.{app_name}"
        print(f"Retrying with: {qualified_app_name}")
        
        # Create new TruApp with fully qualified name
        tru_llm_app_qualified = TruApp(
            llm_app,
            app_name=qualified_app_name,
            app_version=app_version,
            connector=tru_snowflake_connector
        )
        
        # Retry run with qualified name
        run_qualified = tru_llm_app_qualified.add_run(run_config=run_config)
        run_qualified.start()
        run = run_qualified  # Use the successful run for metrics
        print("✅ Run completed with fully qualified name!")

print("Run completed. Computing metrics...")

# Compute evaluation metrics
run.compute_metrics([
    "answer_relevance",
    "groundedness",
])

print("✅ Evaluation completed!")
print(f"Check results in Snowsight: AI & ML > Evaluations > {app_name}")

# Show different ways to reference the external agent
print(f"\n=== External Agent Naming Options ===")
print(f"Simple name: {app_name}")
print(f"Fully qualified: {SNOWFLAKE_CONFIG['database']}.{SNOWFLAKE_CONFIG['schema']}.{app_name}")
print(f"Current context: {session.get_current_database()}.{session.get_current_schema()}")

# Verify external agent creation with both naming approaches
print(f"\n=== Checking External Agents ===")
try:
    # Try simple name first
    simple_check = session.sql(f"DESC EXTERNAL AGENT {app_name}").collect()
    print(f"✅ Found with simple name: {app_name}")
except:
    try:
        # Try fully qualified name
        qualified_name = f"{SNOWFLAKE_CONFIG['database']}.{SNOWFLAKE_CONFIG['schema']}.{app_name}"
        qualified_check = session.sql(f"DESC EXTERNAL AGENT {qualified_name}").collect()
        print(f"✅ Found with qualified name: {qualified_name}")
    except Exception as e:
        print(f"❌ External agent not found with either naming: {e}")

# Show all external agents
print(f"\n=== All External Agents ===")
try:
    all_agents = session.sql("SHOW EXTERNAL AGENTS").collect()
    for agent in all_agents:
        print(f"Agent: {agent}")
    if not all_agents:
        print("No external agents found")
except Exception as e:
    print(f"Error listing external agents: {e}")

# Optional: Check external agent was created
print("\n=== Checking External Agents ===")
external_agents = session.sql("SHOW EXTERNAL AGENTS").collect()
for agent in external_agents:
    print(f"External Agent: {agent}")

# Optional: Check if data was ingested into event table
print("\n=== Checking Event Table ===")
try:
    event_count = session.sql("""
        SELECT COUNT(*) as event_count 
        FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
        WHERE application_name = ?
    """, params=[app_name]).collect()
    print(f"Events in table: {event_count}")
except Exception as e:
    print(f"Error checking event table: {e}")

print("\n=== Setup Verification ===")
print("1. Check if external agent exists: SHOW EXTERNAL AGENTS;")
print("2. Check observability data: SELECT * FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS;")
print("3. View results in Snowsight: AI & ML → Evaluations")
print(f"4. Current context: Role={SNOWFLAKE_CONFIG['role']}, DB={SNOWFLAKE_CONFIG['database']}, Schema={SNOWFLAKE_CONFIG['schema']}")
