import os
from snowflake.snowpark.session import Session
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
    'database': 'your_database',      # Any database you have access to
    'schema': 'your_schema',          # Any schema you have access to
    'role': 'your_role'               # Your role
}

class SimpleAIApp:
    """
    Simple AI application for testing observability - Following official Snowflake AI Observability patterns
    """
    
    def __init__(self, session: Session):
        self.session = session
    
    @instrument(
        span_type=SpanAttributes.SpanType.GENERATION,
        attributes={
            # Map function parameters to official Snowflake AI Observability attributes
            "RECORD_ROOT.INPUT": lambda self, question: question,
            "RECORD_ROOT.OUTPUT": lambda result: result,
            # Additional metadata
            "llm.request.model": "llama3.1-70b",
            "llm.provider": "snowflake-cortex"
        }
    )
    def ask_question(self, question: str) -> str:
        """
        Simple AI question answering using Snowflake Cortex
        Maps to RECORD_ROOT.INPUT and RECORD_ROOT.OUTPUT attributes
        """
        prompt = f"""
        You are a helpful assistant. Answer the following question concisely:
        
        Question: {question}
        
        Answer:
        """
        
        try:
            # Use Snowflake Cortex Complete function
            response = complete("llama3.1-70b", prompt)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            # Map retrieval parameters to official Snowflake AI Observability attributes
            "RETRIEVAL.QUERY_TEXT": lambda self, query: query,
            "RETRIEVAL.RETRIEVED_CONTEXTS": lambda result: result,
            # Additional metadata
            "retrieval.source": "mock_database"
        }
    )
    def retrieve_context(self, query: str) -> list:
        """
        Example retrieval function for RAG applications
        Maps to RETRIEVAL.QUERY_TEXT and RETRIEVAL.RETRIEVED_CONTEXTS attributes
        """
        # This would normally connect to a search service or vector database
        # For demo purposes, return mock context
        mock_contexts = [
            f"Context 1 related to: {query}",
            f"Context 2 about: {query}",
            f"Additional context for: {query}"
        ]
        return mock_contexts
    
    @instrument(
        span_type=SpanAttributes.SpanType.GENERATION,
        attributes={
            # For RAG applications, map both input/output AND retrieval attributes
            "RECORD_ROOT.INPUT": lambda self, question, context=None: question,
            "RECORD_ROOT.OUTPUT": lambda result: result,
            "RETRIEVAL.QUERY_TEXT": lambda self, question, context=None: question,
            # Additional metadata
            "llm.request.model": "llama3.1-70b",
            "llm.provider": "snowflake-cortex",
            "application.type": "rag"
        }
    )
    def ask_question_with_context(self, question: str, context: list = None) -> str:
        """
        RAG-style question answering with context
        Maps to multiple official Snowflake AI Observability attributes for RAG evaluation
        """
        if context is None:
            context = self.retrieve_context(question)
        
        context_str = "\n".join(context)
        prompt = f"""
        Based on the following context, answer the question:
        
        Context:
        {context_str}
        
        Question: {question}
        
        Answer:
        """
        
        try:
            response = complete("llama3.1-70b", prompt)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

def test_basic_cortex(session: Session):
    """
    Test basic Cortex functionality without instrumentation
    """
    print("Testing basic Cortex functionality...")
    try:
        response = complete("llama3.1-70b", "What is 2+2?")
        print("✅ Basic Cortex test successful")
        print(f"   Response: {response}")
        return True
    except Exception as e:
        print(f"❌ Basic Cortex test failed: {e}")
        return False

def main():
    """
    Main function to test AI Observability setup following official Snowflake patterns
    """
    print("🚀 Official Snowflake AI Observability Test (Following Documentation)...")
    print("=" * 80)
    
    # Step 1: Create Snowflake session
    print("\n1. Creating Snowflake session...")
    try:
        session = Session.builder.configs(SNOWFLAKE_CONFIG).create()
        print("✅ Snowflake session created successfully")
        print(f"   Current role: {session.get_current_role()}")
        print(f"   Current database: {session.get_current_database()}")
        print(f"   Current schema: {session.get_current_schema()}")
        
    except Exception as e:
        print(f"❌ Failed to create Snowflake session: {e}")
        return
    
    # Step 2: Test basic Cortex functionality
    print("\n2. Testing basic Cortex functionality...")
    if not test_basic_cortex(session):
        print("❌ Cannot proceed without basic Cortex access")
        return
    
    # Step 3: Create AI application
    print("\n3. Creating AI application...")
    try:
        ai_app = SimpleAIApp(session)
        print("✅ AI application created")
    except Exception as e:
        print(f"❌ Failed to create AI application: {e}")
        return
    
    # Step 4: Setup TruLens connector
    print("\n4. Setting up TruLens connector...")
    try:
        tru_snowflake_connector = SnowflakeConnector(snowpark_session=session)
        print("✅ TruLens Snowflake connector created")
    except Exception as e:
        print(f"❌ Failed to create TruLens connector: {e}")
        return
    
    # Step 5: Create TruApp for observability
    print("\n5. Creating TruApp for observability...")
    try:
        app_name = "official_test_app"
        app_version = "v1.0"
        
        tru_app = TruApp(
            ai_app,
            app_name=app_name,
            app_version=app_version,
            connector=tru_snowflake_connector
        )
        print("✅ TruApp created successfully")
        print(f"   App name: {app_name}")
        print(f"   App version: {app_version}")
    except Exception as e:
        print(f"❌ Failed to create TruApp: {e}")
        return
    
    # Step 6: Test different types of instrumented AI calls
    print("\n6. Testing instrumented AI calls with official Snowflake attributes...")
    try:
        # Test simple generation (RECORD_ROOT.INPUT/OUTPUT)
        print("   Testing simple generation with RECORD_ROOT attributes...")
        with tru_app as recording:
            response1 = ai_app.ask_question("What is artificial intelligence?")
        print(f"   ✅ Simple generation completed: {response1[:100]}...")
        
        # Test retrieval (RETRIEVAL.QUERY_TEXT/RETRIEVED_CONTEXTS)
        print("   Testing retrieval with RETRIEVAL attributes...")
        with tru_app as recording:
            contexts = ai_app.retrieve_context("Snowflake features")
        print(f"   ✅ Retrieval completed: {len(contexts)} contexts retrieved")
        
        # Test RAG-style generation (combines both attribute types)
        print("   Testing RAG generation with combined attributes...")
        with tru_app as recording:
            response2 = ai_app.ask_question_with_context(
                "What are the benefits of machine learning?",
                context=[
                    "Machine learning helps automate decision making",
                    "ML can process large amounts of data quickly",
                    "Machine learning improves over time with more data"
                ]
            )
        print(f"   ✅ RAG generation completed: {response2[:100]}...")
        
        print("✅ All instrumented calls completed with official Snowflake attributes")
        
    except Exception as e:
        print(f"❌ Instrumented calls failed: {e}")
        return
    
    print("\n" + "=" * 80)
    print("🎉 Official Snowflake AI Observability test completed!")
    print("\nImplementation follows official Snowflake AI Observability documentation:")
    print("✓ Uses official reserved attributes from Snowflake docs")
    print("✓ RECORD_ROOT.INPUT/OUTPUT for generation spans")
    print("✓ RETRIEVAL.QUERY_TEXT/RETRIEVED_CONTEXTS for retrieval spans")
    print("✓ Proper lambda functions for parameter mapping")
    print("✓ Compatible with Snowflake AI Observability evaluation metrics")
    
    print("\nOfficial Snowflake AI Observability attributes used:")
    print("- RECORD_ROOT.INPUT: Input prompt to the LLM")
    print("- RECORD_ROOT.OUTPUT: Generated response from the LLM")
    print("- RETRIEVAL.QUERY_TEXT: User query for RAG application")
    print("- RETRIEVAL.RETRIEVED_CONTEXTS: Context retrieved from search service")
    
    print("\nSupported evaluation metrics with this implementation:")
    print("- Answer Relevance: Uses RECORD_ROOT.INPUT + RECORD_ROOT.OUTPUT")
    print("- Context Relevance: Uses RETRIEVAL.QUERY_TEXT + RETRIEVAL.RETRIEVED_CONTEXTS")
    print("- Groundedness: Uses RETRIEVAL.RETRIEVED_CONTEXTS + RECORD_ROOT.OUTPUT")
    print("- Coherence: Uses RECORD_ROOT.OUTPUT")
    print("- Cost and Latency: Automatically tracked")
    
    print("\nNext steps:")
    print("1. Run this code and verify no parameter mapping errors")
    print("2. Query: SELECT * FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS")
    print("3. Check for non-null TRACE_ID and proper ATTRIBUTES")
    print("4. Go to Snowsight → AI & ML → Evaluations to see your app")
    print("5. Create datasets and runs for evaluation")
    
    # Cleanup
    session.close()

if __name__ == "__main__":
    print("Required packages (latest versions):")
    print("- snowflake-snowpark-python")
    print("- trulens-core")
    print("- trulens-providers-cortex")
    print("- trulens-connectors-snowflake")
    print("\nInstall with:")
    print("pip install snowflake-snowpark-python trulens-core trulens-providers-cortex trulens-connectors-snowflake")
    print("\n" + "=" * 80)
    
    main()
