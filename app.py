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
    Simple AI application for testing observability with proper instrumentation
    """
    
    def __init__(self, session: Session):
        self.session = session
    
    @instrument(
        span_type=SpanAttributes.SpanType.GENERATION,
        attributes={
            # Use string keys instead of SpanAttributes constants for better compatibility
            "llm.request.model": "llama3.1-70b",
            "llm.provider": "snowflake-cortex",
            "llm.request.type": "completion"
        }
    )
    def ask_question(self, question: str) -> str:
        """
        Simple AI question answering using Snowflake Cortex
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
            "retrieval.source": "mock_database",
            "retrieval.type": "similarity_search"
        }
    )
    def retrieve_context(self, query: str) -> list:
        """
        Example retrieval function for RAG applications
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
            "llm.request.model": "llama3.1-70b",
            "llm.provider": "snowflake-cortex",
            "llm.request.type": "rag_completion",
            "application.type": "question_answering"
        }
    )
    def ask_question_with_context(self, question: str, context: list = None) -> str:
        """
        RAG-style question answering with context
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
    Main function to test AI Observability setup with fixed instrumentation
    """
    print("🚀 Fixed Snowflake AI Observability Test...")
    print("=" * 70)
    
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
        app_name = "fixed_test_app"
        app_version = "v1.2"
        
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
    print("\n6. Testing instrumented AI calls with fixed instrumentation...")
    try:
        # Test simple generation
        print("   Testing simple generation...")
        with tru_app as recording:
            response1 = ai_app.ask_question("What is artificial intelligence?")
        print(f"   ✅ Simple generation completed: {response1[:100]}...")
        
        # Test RAG-style generation
        print("   Testing RAG-style generation...")
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
        
        # Test retrieval function
        print("   Testing retrieval instrumentation...")
        with tru_app as recording:
            contexts = ai_app.retrieve_context("Snowflake features")
        print(f"   ✅ Retrieval completed: {len(contexts)} contexts retrieved")
        
        print("✅ All instrumented calls completed successfully with fixed attributes")
        
    except Exception as e:
        print(f"❌ Instrumented calls failed: {e}")
        return
    
    print("\n" + "=" * 70)
    print("🎉 Fixed AI Observability test completed!")
    print("\nKey fixes made:")
    print("✓ Removed problematic lambda functions from span attributes")
    print("✓ Used simple string-based attribute keys for better compatibility")
    print("✓ Simplified @instrument() decorators to avoid parameter mapping issues")
    print("✓ Added relevant LLM and application metadata")
    
    print("\nFixed Instrumentation approach:")
    print("- @instrument() with span_type and static attributes")
    print("- No complex lambda functions that cause parameter access errors")
    print("- Simple string keys for attributes (e.g., 'llm.request.model')")
    print("- Proper span types: GENERATION and RETRIEVAL")
    
    print("\nNext steps:")
    print("1. Run this code and check for errors")
    print("2. Query: SELECT * FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS")
    print("3. Look for non-null TRACE_ID and SPAN_ID values")
    print("4. Check ATTRIBUTES column for your custom metadata")
    
    # Cleanup
    session.close()

if __name__ == "__main__":
    print("Required packages:")
    print("- snowflake-snowpark-python")
    print("- trulens-core")
    print("- trulens-providers-cortex")
    print("- trulens-connectors-snowflake")
    print("\nInstall with:")
    print("pip install snowflake-snowpark-python trulens-core trulens-providers-cortex trulens-connectors-snowflake")
    print("\n" + "=" * 70)
    
    main()
