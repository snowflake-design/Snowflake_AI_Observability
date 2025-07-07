import os
import time
import logging
from snowflake.snowpark.session import Session
from snowflake.cortex import complete
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
from trulens.apps.app import TruApp
from trulens.connectors.snowflake import SnowflakeConnector

# --- Configure Logging (Reduced Verbosity) ---
# Set the root logger to INFO to see general progress
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# Suppress overly verbose DEBUG messages from specific libraries
logging.getLogger('snowflake.snowpark').setLevel(logging.WARNING)
logging.getLogger('trulens').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING) # Often noisy from HTTP requests

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
# This activates OpenTelemetry collection within TruLens
os.environ["TRULENS_OTEL_TRACING"] = "1"

# Define your AI application class
class SimpleAIApp:
    def __init__(self, session: Session):
        self.session = session
    
    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate_answer(self, question: str) -> str:
        """
        Simple AI question answering using Snowflake Cortex complete function.
        """
        logging.info(f"  [AIApp] Generating answer for: '{question}'...")
        prompt = f"""
        You are a helpful assistant. Answer the following question concisely:
        
        Question: {question}
        
        Answer:
        """
        try:
            # Use Snowflake Cortex Complete function (model name should be available in your region)
            response = complete("mixtral-8x7b", prompt) # Changed to mixtral-8x7b, common and available
            logging.info(f"  [AIApp] Answer generated.")
            return response
        except Exception as e:
            logging.error(f"  [AIApp] Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
            
    @instrument(span_type=SpanAttributes.SpanType.RETRIEVAL)
    def retrieve_info(self, topic: str) -> list:
        """
        Simulated retrieval function. In a real RAG app, this would query a vector DB.
        """
        logging.info(f"  [AIApp] Retrieving info for: '{topic}'...")
        # Simulate a delay and return mock data
        time.sleep(0.5) 
        mock_info = [
            f"Fact 1: Snowflake is a cloud data platform.",
            f"Fact 2: It offers data warehousing, data lakes, and data engineering capabilities.",
            f"Fact 3: Snowflake Cortex provides AI/ML functions, including LLMs.",
            f"Fact 4: AI Observability helps monitor AI applications."
        ]
        relevant_info = [info for info in mock_info if topic.lower() in info.lower()]
        logging.info(f"  [AIApp] Retrieved {len(relevant_info)} pieces of info.")
        return relevant_info if relevant_info else mock_info[:2] # Return some info even if topic not found

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def ask_with_context(self, question: str, context_list: list) -> str:
        """
        RAG-style question answering that uses provided context.
        """
        logging.info(f"  [AIApp] Generating answer with context for: '{question}'...")
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
            logging.info(f"  [AIApp] Answer with context generated.")
            return response
        except Exception as e:
            logging.error(f"  [AIApp] Error generating response with context: {str(e)}")
            return f"Error generating response with context: {str(e)}"

def main():
    logging.info("--- Snowflake AI Observability Test Script ---")
    logging.info("1. Ensuring Snowflake session is created.")
    session = None
    try:
        session = Session.builder.configs(SNOWFLAKE_CONFIG).create()
        logging.info(f"✅ Session created successfully. Current role: {session.get_current_role()}")
        logging.info(f"   Database: {session.get_current_database()}, Schema: {session.get_current_schema()}")
        
        # Explicitly set current context (important for stability)
        session.use_database(SNOWFLAKE_CONFIG['database'])
        session.use_schema(SNOWFLAKE_CONFIG['schema'])
        session.use_warehouse(SNOWFLAKE_CONFIG['warehouse'])
        session.use_role(SNOWFLAKE_CONFIG['role'])
        logging.info("✅ Session context explicitly set.")

    except Exception as e:
        logging.error(f"❌ Failed to create Snowflake session: {e}")
        logging.error("Please check your SNOWFLAKE_CONFIG details, network connectivity, and ensure necessary database/schema usage grants.")
        return

    logging.info("\n2. Initializing AI Application.")
    ai_app = SimpleAIApp(session)
    logging.info("✅ AI Application initialized.")

    logging.info("\n3. Setting up TruLens Snowflake Connector.")
    tru_connector = None
    try:
        tru_connector = SnowflakeConnector(snowpark_session=session)
        logging.info("✅ TruLens Snowflake Connector created.")
    except Exception as e:
        logging.error(f"❌ Failed to create TruLens Snowflake Connector: {e}")
        logging.error("Ensure 'trulens-connectors-snowflake' is installed.")
        session.close()
        return

    APP_NAME = "TESTAPP" # Changed for a fresh start!
    APP_VERSION = "v1.0"

    logging.info(f"\n4. Creating TruApp for observability (Name: {APP_NAME}, Version: {APP_VERSION}).")
    tru_app = None
    try:
        tru_app = TruApp(
            ai_app,
            app_name=APP_NAME,
            app_version=APP_VERSION,
            connector=tru_connector
        )
        logging.info("✅ TruApp created successfully.")
        logging.info(f"   Note: This step attempts to create/verify the EXTERNAL AGENT '{APP_NAME}' VERSION '{APP_VERSION}' in Snowflake.")
        logging.info("   If this step fails, check your role's permissions (SNOWFLAKE.AI_OBSERVABILITY_ADMIN) and ensure no existing agent with this exact name/version is blocking it.")
    except Exception as e:
        logging.error(f"❌ Failed to create TruApp: {e}")
        logging.error("This is where the EXTERNAL AGENT is registered/created. A failure here will likely lead to ingestion errors.")
        logging.error(f"Detailed Error: {e}") # Keep this for debugging the TruApp creation itself
        session.close()
        return

    logging.info("\n5. Running instrumented AI calls to generate observability data.")
    
    # Test a simple generation
    logging.info("\n--- Test Case 1: Simple LLM Generation ---")
    try:
        with tru_app as recording:
            question1 = "What is the capital of France?"
            answer1 = ai_app.generate_answer(question1)
            logging.info(f"Response: {answer1[:100]}...")
    except Exception as e:
        logging.error(f"Error during simple generation test: {e}")

    # Test a RAG-style flow (retrieve then generate)
    logging.info("\n--- Test Case 2: RAG-style Flow ---")
    try:
        with tru_app as recording:
            topic_query = "Snowflake observability"
            context_retrieved = ai_app.retrieve_info(topic_query)
            rag_question = "What does Snowflake offer for AI monitoring?"
            rag_answer = ai_app.ask_with_context(rag_question, context_retrieved)
            logging.info(f"Response: {rag_answer[:100]}...")
    except Exception as e:
        logging.error(f"Error during RAG-style test: {e}")

    logging.info("\n--- All instrumented calls completed. ---")
    logging.info("TruLens is now attempting to upload data to Snowflake (please wait for logs).")
    time.sleep(10) # Give TruLens some time to upload data asynchronously

    logging.info("\n6. Verification Steps in Snowflake (after a few minutes):")
    logging.info("   a. Go to Snowsight -> AI & ML -> Evaluations (under 'Observability').")
    logging.info(f"      You should see your application '{APP_NAME}' (Version: {APP_VERSION}) listed.")
    logging.info("   b. Query the event table directly (wait 5-15 minutes for data propagation):")
    logging.info(f"      SELECT * FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS")
    logging.info(f"      WHERE APPLICATION_NAME = '{APP_NAME}' AND APPLICATION_VERSION = '{APP_VERSION}';")
    logging.info("      Check if TRACE_ID, SPAN_ID, and data in the 'RECORD' column are populated (not NULL).")

    logging.info("\n7. Closing Snowflake session.")
    try:
        if session:
            session.close()
            logging.info("✅ Snowflake session closed successfully.")
    except Exception as e:
        logging.warning(f"ℹ️ Error closing session (might be already closed or a minor cleanup issue): {e}")

if __name__ == "__main__":
    main()
