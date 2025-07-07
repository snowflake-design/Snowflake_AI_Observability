import os
import time
from snowflake.snowpark.session import Session
from snowflake.cortex import complete
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
from trulens.apps.app import TruApp
from trulens.connectors.snowflake import SnowflakeConnector

# Add more logging for debugging
import logging
logging.basicConfig(level=logging.DEBUG) # Set to DEBUG for maximum verbosity
logging.getLogger('snowflake.snowpark').setLevel(logging.INFO) # Adjust Snowpark level if too noisy
logging.getLogger('trulens').setLevel(logging.DEBUG)

# --- IMPORTANT: Configure your Snowflake connection details ---
SNOWFLAKE_CONFIG = {
    'account': os.getenv('SNOWFLAKE_ACCOUNT', 'YOUR_SNOWFLAKE_ACCOUNT_IDENTIFIER'), # e.g., 'xyz12345.us-east-1'
    'user': os.getenv('SNOWFLAKE_USER', 'YOUR_SNOWFLAKE_USERNAME'),
    'password': os.getenv('SNOWFLAKE_PASSWORD', 'YOUR_SNOWFLAKE_PASSWORD'),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE', 'YOUR_SNOWFLAKE_WAREHOUSE'),
    'database': os.getenv('SNOWFLAKE_DATABASE', 'YOUR_SNOWFLAKE_DATABASE'),
    'schema': os.getenv('SNOWFLAKE_SCHEMA', 'YOUR_SNOWFLAKE_SCHEMA'),
    'role': os.getenv('SNOWFLAKE_ROLE', 'YOUR_SNOWFLAKE_ROLE') # This role MUST have SNOWFLAKE.AI_OBSERVABILITY_ADMIN
}

# Set environment variable for TruLens OpenTelemetry tracing
os.environ["TRULENS_OTEL_TRACING"] = "1"

# Define your AI application class
class SimpleAIApp:
    def __init__(self, session: Session):
        self.session = session
    
    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate_answer(self, question: str) -> str:
        print(f"  [AIApp] Generating answer for: '{question}'...")
        prompt = f"""
        You are a helpful assistant. Answer the following question concisely:
        
        Question: {question}
        
        Answer:
        """
        try:
            response = complete("mixtral-8x7b", prompt)
            print(f"  [AIApp] Answer generated.")
            return response
        except Exception as e:
            print(f"  [AIApp] Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
            
    @instrument(span_type=SpanAttributes.SpanType.RETRIEVAL)
    def retrieve_info(self, topic: str) -> list:
        print(f"  [AIApp] Retrieving info for: '{topic}'...")
        time.sleep(0.5) 
        mock_info = [
            f"Fact 1: Snowflake is a cloud data platform.",
            f"Fact 2: It offers data warehousing, data lakes, and data engineering capabilities.",
            f"Fact 3: Snowflake Cortex provides AI/ML functions, including LLMs.",
            f"Fact 4: AI Observability helps monitor AI applications."
        ]
        relevant_info = [info for info in mock_info if topic.lower() in info.lower()]
        print(f"  [AIApp] Retrieved {len(relevant_info)} pieces of info.")
        return relevant_info if relevant_info else mock_info[:2]

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def ask_with_context(self, question: str, context_list: list) -> str:
        print(f"  [AIApp] Generating answer with context for: '{question}'...")
        context_str = "\n".join(context_list)
        prompt = f"""
        Based on the following context, answer the question concisely.
        
        Context:
        {context_str}
        
        Question: {question}
        
        Answer:
        """
        try:
            response = complete("mixtral-8x7b", prompt)
            print(f"  [AIApp] Answer with context generated.")
            return response
        except Exception as e:
            print(f"  [AIApp] Error generating response with context: {str(e)}")
            return f"Error generating response with context: {str(e)}"

def main():
    print("--- Snowflake AI Observability Test Script ---")
    print("1. Ensuring Snowflake session is created.")
    session = None
    try:
        session = Session.builder.configs(SNOWFLAKE_CONFIG).create()
        print(f"✅ Session created successfully. Current role: {session.get_current_role()}")
        print(f"   Database: {session.get_current_database()}, Schema: {session.get_current_schema()}")
    except Exception as e:
        print(f"❌ Failed to create Snowflake session: {e}")
        print("Please check your SNOWFLAKE_CONFIG details and network connectivity.")
        return

    print("\n2. Initializing AI Application.")
    ai_app = SimpleAIApp(session)
    print("✅ AI Application initialized.")

    print("\n3. Setting up TruLens Snowflake Connector.")
    tru_connector = None
    try:
        tru_connector = SnowflakeConnector(snowpark_session=session)
        print("✅ TruLens Snowflake Connector created.")
    except Exception as e:
        print(f"❌ Failed to create TruLens Snowflake Connector: {e}")
        print("Ensure 'trulens-connectors-snowflake' is installed.")
        session.close()
        return

    APP_NAME = "MySimpleAIApp" 
    APP_VERSION = "v1.0"

    print(f"\n4. Creating TruApp for observability (Name: {APP_NAME}, Version: {APP_VERSION}).")
    tru_app = None
    try:
        tru_app = TruApp(
            ai_app,
            app_name=APP_NAME,
            app_version=APP_VERSION,
            connector=tru_connector
        )
        print("✅ TruApp created successfully.")
        print(f"   Note: This step attempts to create/verify the EXTERNAL AGENT '{APP_NAME}' VERSION '{APP_VERSION}' in Snowflake.")
    except Exception as e:
        print(f"❌ Failed to create TruApp: {e}")
        print("This is where the EXTERNAL AGENT is registered/created. A failure here will lead to ingestion errors.")
        print(f"Detailed Error: {e}") # Print the full exception here
        session.close()
        return

    print("\n5. Running instrumented AI calls to generate observability data.")
    
    # Test a simple generation
    print("\n--- Test Case 1: Simple LLM Generation ---")
    with tru_app as recording:
        question1 = "What is the capital of France?"
        answer1 = ai_app.generate_answer(question1)
        print(f"Response: {answer1[:100]}...")

    # Test a RAG-style flow (retrieve then generate)
    print("\n--- Test Case 2: RAG-style Flow ---")
    with tru_app as recording:
        topic_query = "Snowflake observability"
        context_retrieved = ai_app.retrieve_info(topic_query)
        rag_question = "What does Snowflake offer for AI monitoring?"
        rag_answer = ai_app.ask_with_context(rag_question, context_retrieved)
        print(f"Response: {rag_answer[:100]}...")

    print("\n--- All instrumented calls completed. ---")
    print("TruLens is now attempting to upload data to Snowflake (please wait for logs).")
    time.sleep(10) # Give TruLens some time to upload data asynchronously

    print("\n6. Verification Steps in Snowflake (after a few minutes):")
    print("   a. Go to Snowsight -> AI & ML -> Evaluations (under 'Observability').")
    print(f"      You should see your application '{APP_NAME}' (Version: {APP_VERSION}) listed.")
    print("   b. Query the event table directly (wait 5-15 minutes for data propagation):")
    print(f"      SELECT * FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS")
    print(f"      WHERE APPLICATION_NAME = '{APP_NAME}' AND APPLICATION_VERSION = '{APP_VERSION}';")
    print("      Check if TRACE_ID, SPAN_ID, and data in the 'RECORD' column are populated (not NULL).")

    print("\n7. Closing Snowflake session.")
    try:
        if session:
            session.close()
            print("✅ Snowflake session closed successfully.")
    except Exception as e:
        print(f"ℹ️ Error closing session (might be already closed): {e}")

if __name__ == "__main__":
    main()
