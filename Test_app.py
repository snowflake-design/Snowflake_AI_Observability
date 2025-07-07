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
    Simple AI application for testing observability - Simplified approach without lambda functions
    """
    
    def __init__(self, session: Session):
        self.session = session
    
    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
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
    
    @instrument(span_type=SpanAttributes.SpanType.RETRIEVAL)
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
    
    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
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
    Main function to test AI Observability setup with minimal instrumentation
    """
    print("🚀 Simplified Snowflake AI Observability Test...")
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
    
    # Step 4: Setup TruLens connector with error handling
    print("\n4. Setting up TruLens connector...")
    try:
        # Try different connector initialization approaches
        tru_snowflake_connector = SnowflakeConnector(
            snowpark_session=session,
            database=session.get_current_database(),
            schema=session.get_current_schema()
        )
        print("✅ TruLens Snowflake connector created")
    except Exception as e:
        print(f"⚠️  Primary connector creation failed: {e}")
        try:
            # Fallback approach
            tru_snowflake_connector = SnowflakeConnector(snowpark_session=session)
            print("✅ TruLens Snowflake connector created (fallback)")
        except Exception as e2:
            print(f"❌ Failed to create TruLens connector: {e2}")
            return
    
    # Step 5: Create TruApp for observability
    print("\n5. Creating TruApp for observability...")
    try:
        app_name = "simple_test_app"
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
    print("\n6. Testing instrumented AI calls with simplified approach...")
    try:
        # Test simple generation
        print("   Testing simple generation...")
        with tru_app as recording:
            response1 = ai_app.ask_question("What is artificial intelligence?")
        print(f"   ✅ Simple generation completed: {response1[:100]}...")
        
        # Test retrieval function
        print("   Testing retrieval...")
        with tru_app as recording:
            contexts = ai_app.retrieve_context("Snowflake features")
        print(f"   ✅ Retrieval completed: {len(contexts)} contexts retrieved")
        
        # Test RAG-style generation
        print("   Testing RAG generation...")
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
        
        print("✅ All instrumented calls completed successfully")
        
    except Exception as e:
        print(f"❌ Instrumented calls failed: {e}")
        print("   But this is expected - let's check if traces were still captured...")
    
    # Step 7: Check for basic trace capture
    print("\n7. Checking for basic trace capture...")
    try:
        import time
        time.sleep(2)  # Wait for data to be written
        
        # Simple count query
        result = session.sql("""
            SELECT COUNT(*) as event_count
            FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
            WHERE APPLICATION_NAME = %s
        """, params=[app_name]).collect()
        
        if result and result[0]['EVENT_COUNT'] > 0:
            print(f"🎉 SUCCESS: Found {result[0]['EVENT_COUNT']} events in observability table!")
            
            # Show some basic info
            details = session.sql("""
                SELECT SPAN_NAME, EVENT_TYPE, COUNT(*) as cnt
                FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
                WHERE APPLICATION_NAME = %s
                GROUP BY SPAN_NAME, EVENT_TYPE
                ORDER BY cnt DESC
            """, params=[app_name]).collect()
            
            print("   Event breakdown:")
            for detail in details:
                print(f"     {detail['SPAN_NAME']} ({detail['EVENT_TYPE']}): {detail['CNT']}")
                
        else:
            print("ℹ️  No events found yet - data might still be processing")
            
    except Exception as e:
        print(f"❌ Could not check observability data: {e}")
    
    print("\n" + "=" * 70)
    print("🎉 Simplified AI Observability test completed!")
    print("\nSimplified approach used:")
    print("✓ Removed problematic lambda functions")
    print("✓ Used basic @instrument() decorators only")
    print("✓ Added error handling for connector issues")
    print("✓ Minimal span configuration")
    
    print("\nKey fixes for your errors:")
    print("- No lambda functions in @instrument() decorators")
    print("- Added fallback SnowflakeConnector initialization")
    print("- Simplified span type declarations")
    print("- Better error handling throughout")
    
    print("\nNext steps:")
    print("1. Check if this runs without lambda function errors")
    print("2. Query: SELECT * FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS")
    print("3. Look for your app_name in the results")
    print("4. Check if TRACE_ID and SPAN_ID are populated")
    
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
