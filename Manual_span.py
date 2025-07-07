import os
import time
import logging
from snowflake.snowpark.session import Session
from snowflake.cortex import complete

# Import TruLens components in the correct order
from trulens.core import TruSession
from trulens.apps.app import TruApp
from trulens.connectors.snowflake import SnowflakeConnector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.getLogger('snowflake.snowpark').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Snowflake connection parameters
SNOWFLAKE_CONFIG = {
    'account': 'your_account',
    'user': 'your_username', 
    'password': 'your_password',
    'warehouse': 'your_warehouse',
    'database': 'your_database',
    'schema': 'your_schema',
    'role': 'your_role'
}

# CRITICAL: Set TruLens environment variables BEFORE any imports
os.environ["TRULENS_OTEL_TRACING"] = "1"

class SimpleAIApp:
    """
    AI Application without @instrument decorators - we'll handle tracing manually
    """
    
    def __init__(self, session: Session):
        self.session = session
        self.app_name = "manual_trace_app"
    
    def generate_answer(self, question: str) -> str:
        """Simple generation without decorator"""
        logging.info(f"  [AIApp] Generating answer for: '{question[:50]}...'")
        
        prompt = f"""
        You are a helpful assistant. Answer the following question concisely:
        
        Question: {question}
        
        Answer:
        """
        
        try:
            response = complete("mixtral-8x7b", prompt)
            logging.info(f"  [AIApp] Answer generated: {len(response)} chars")
            return response
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logging.error(f"  [AIApp] {error_msg}")
            return error_msg
    
    def retrieve_info(self, topic: str) -> list:
        """Simple retrieval without decorator"""
        logging.info(f"  [AIApp] Retrieving info for: '{topic}'")
        
        mock_info = [
            f"Information about {topic}: This is context item 1",
            f"More details on {topic}: This is context item 2", 
            f"Additional facts about {topic}: This is context item 3"
        ]
        
        logging.info(f"  [AIApp] Retrieved {len(mock_info)} items")
        return mock_info
    
    def ask_with_context(self, question: str, context_list: list) -> str:
        """RAG-style generation without decorator"""
        logging.info(f"  [AIApp] Generating with context for: '{question[:50]}...'")
        
        context_str = "\n".join(context_list)
        prompt = f"""
        Based on the following context, answer the question:
        
        Context:
        {context_str}
        
        Question: {question}
        
        Answer:
        """
        
        try:
            response = complete("mixtral-8x7b", prompt)
            logging.info(f"  [AIApp] Context-based answer generated: {len(response)} chars")
            return response
        except Exception as e:
            error_msg = f"Error generating response with context: {str(e)}"
            logging.error(f"  [AIApp] {error_msg}")
            return error_msg

class InstrumentedAIApp:
    """
    Wrapper class that adds manual instrumentation to ensure traces are captured
    """
    
    def __init__(self, base_app: SimpleAIApp, tru_app: TruApp):
        self.base_app = base_app
        self.tru_app = tru_app
    
    def generate_answer(self, question: str) -> str:
        """Manually instrumented generation"""
        logging.info("🎯 Starting instrumented generation...")
        
        # Manual span creation and context management
        with self.tru_app.tracer.start_as_current_span("generate_answer") as span:
            span.set_attribute("operation.type", "generation")
            span.set_attribute("model.name", "mixtral-8x7b")
            span.set_attribute("input.question", question)
            
            result = self.base_app.generate_answer(question)
            
            span.set_attribute("output.response", result[:100])
            span.set_attribute("output.length", len(result))
            
            logging.info("✅ Manual instrumentation completed")
            return result
    
    def retrieve_info(self, topic: str) -> list:
        """Manually instrumented retrieval"""
        logging.info("🎯 Starting instrumented retrieval...")
        
        with self.tru_app.tracer.start_as_current_span("retrieve_info") as span:
            span.set_attribute("operation.type", "retrieval")
            span.set_attribute("input.topic", topic)
            
            result = self.base_app.retrieve_info(topic)
            
            span.set_attribute("output.count", len(result))
            span.set_attribute("output.items", str(result))
            
            logging.info("✅ Manual retrieval instrumentation completed")
            return result
    
    def ask_with_context(self, question: str, context_list: list) -> str:
        """Manually instrumented RAG"""
        logging.info("🎯 Starting instrumented RAG generation...")
        
        with self.tru_app.tracer.start_as_current_span("ask_with_context") as span:
            span.set_attribute("operation.type", "rag_generation")
            span.set_attribute("model.name", "mixtral-8x7b")
            span.set_attribute("input.question", question)
            span.set_attribute("input.context_count", len(context_list))
            
            result = self.base_app.ask_with_context(question, context_list)
            
            span.set_attribute("output.response", result[:100])
            span.set_attribute("output.length", len(result))
            
            logging.info("✅ Manual RAG instrumentation completed")
            return result

def main():
    logging.info("=== Manual Instrumentation AI Observability Test ===")
    
    # Step 1: Create Snowflake session
    logging.info("1. Creating Snowflake session...")
    session = None
    try:
        session = Session.builder.configs(SNOWFLAKE_CONFIG).create()
        logging.info(f"✅ Session created. Role: {session.get_current_role()}")
        logging.info(f"   Database: {session.get_current_database()}")
        logging.info(f"   Schema: {session.get_current_schema()}")
        
    except Exception as e:
        logging.error(f"❌ Failed to create session: {e}")
        return
    
    # Step 2: Initialize TruSession explicitly
    logging.info("2. Initializing TruSession...")
    try:
        tru_session = TruSession()
        logging.info("✅ TruSession initialized")
    except Exception as e:
        logging.error(f"❌ Failed to initialize TruSession: {e}")
        return
    
    # Step 3: Create connector
    logging.info("3. Creating SnowflakeConnector...")
    try:
        connector = SnowflakeConnector(snowpark_session=session)
        logging.info("✅ SnowflakeConnector created")
    except Exception as e:
        logging.error(f"❌ Failed to create connector: {e}")
        return
    
    # Step 4: Create base AI app
    logging.info("4. Creating AI application...")
    base_app = SimpleAIApp(session)
    logging.info("✅ Base AI application created")
    
    # Step 5: Create TruApp
    logging.info("5. Creating TruApp...")
    APP_NAME = "manual_instrumentation_test"
    APP_VERSION = "v1.0"
    
    try:
        tru_app = TruApp(
            base_app,
            app_name=APP_NAME,
            app_version=APP_VERSION,
            connector=connector
        )
        logging.info(f"✅ TruApp created: {APP_NAME} v{APP_VERSION}")
    except Exception as e:
        logging.error(f"❌ Failed to create TruApp: {e}")
        return
    
    # Step 6: Create manually instrumented wrapper
    logging.info("6. Creating manually instrumented wrapper...")
    instrumented_app = InstrumentedAIApp(base_app, tru_app)
    logging.info("✅ Instrumented wrapper created")
    
    # Step 7: Run tests with manual instrumentation
    logging.info("7. Running manually instrumented tests...")
    
    try:
        # Test 1: Simple generation
        logging.info("\n--- Test 1: Simple Generation ---")
        with tru_app as recording:
            answer1 = instrumented_app.generate_answer("What is artificial intelligence?")
            logging.info(f"Generated: {answer1[:50]}...")
        
        # Test 2: Retrieval
        logging.info("\n--- Test 2: Information Retrieval ---")
        with tru_app as recording:
            contexts = instrumented_app.retrieve_info("machine learning")
            logging.info(f"Retrieved {len(contexts)} contexts")
        
        # Test 3: RAG generation
        logging.info("\n--- Test 3: RAG Generation ---")
        with tru_app as recording:
            rag_answer = instrumented_app.ask_with_context(
                "How does machine learning work?",
                contexts
            )
            logging.info(f"RAG answer: {rag_answer[:50]}...")
        
        logging.info("\n✅ All manual instrumentation tests completed")
        
    except Exception as e:
        logging.error(f"❌ Manual instrumentation tests failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 8: Wait for data processing
    logging.info("\n8. Waiting for trace data to be processed...")
    time.sleep(10)
    
    logging.info("\n=== VERIFICATION STEPS ===")
    logging.info("Wait 5-15 minutes, then run:")
    logging.info(f"""
    SELECT 
        APPLICATION_NAME,
        APPLICATION_VERSION,
        OBSERVED_TIMESTAMP,
        TRACE:trace_id::string as TRACE_ID,
        TRACE:span_id::string as SPAN_ID,
        RECORD:name::string as SPAN_NAME,
        RECORD:attributes::string as ATTRIBUTES
    FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
    WHERE APPLICATION_NAME = '{APP_NAME}'
    ORDER BY OBSERVED_TIMESTAMP DESC;
    """)
    
    logging.info("\nIf traces are still null, the issue is likely:")
    logging.info("1. TruLens version compatibility")
    logging.info("2. OpenTelemetry configuration")
    logging.info("3. Snowflake permissions for AI_OBSERVABILITY_EVENTS table")
    
    # Cleanup
    logging.info("\n9. Cleaning up...")
    try:
        time.sleep(5)  # Give TruLens time to finish
        session.close()
        logging.info("✅ Session closed")
    except Exception as e:
        logging.warning(f"Session cleanup: {e}")

if __name__ == "__main__":
    main()
