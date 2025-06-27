

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
    Simple AI application for testing observability
    """
    
    def __init__(self, session: Session):
        self.session = session
    
    @instrument(
        span_type=SpanAttributes.SpanType.GENERATION,
        attributes={
            SpanAttributes.GENERATION.INPUT: "prompt",
            SpanAttributes.GENERATION.OUTPUT: "return",
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
    Main function to test AI Observability setup minimally
    """
    print("🚀 Minimal Snowflake AI Observability Test...")
    print("=" * 50)
    
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
        app_name = "minimal_test_app"
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
    
    # Step 6: Test instrumented AI calls
    print("\n6. Testing instrumented AI calls...")
    try:
        test_questions = [
            "What is artificial intelligence?",
            "Explain machine learning in simple terms",
            "What is Snowflake?"
        ]
        
        print("Making instrumented AI calls...")
        for i, question in enumerate(test_questions, 1):
            print(f"   Question {i}: {question}")
            
            # This call will be automatically instrumented and traced
            with tru_app as recording:
                response = ai_app.ask_question(question)
            
            print(f"   Response: {response[:100]}...")
            print(f"   ✅ Call {i} completed with tracing")
        
        print("✅ All instrumented calls completed successfully")
        
    except Exception as e:
        print(f"❌ Instrumented calls failed: {e}")
        return
    
    # Step 7: Check for observability data
    print("\n7. Checking for AI Observability data...")
    try:
        # Wait a moment for data to be written
        import time
        time.sleep(2)
        
        # Try to query the observability events table
        result = session.sql("""
            SELECT COUNT(*) as event_count 
            FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
            WHERE APPLICATION_NAME = %s
        """, params=[app_name]).collect()
        
        count = result[0]['EVENT_COUNT']
        print(f"✅ AI Observability Events table accessible")
        print(f"   Events for app '{app_name}': {count}")
        
        if count > 0:
            print("🎉 SUCCESS: Observability data is being captured!")
            
            # Show sample of recent events
            recent_events = session.sql("""
                SELECT EVENT_TIMESTAMP, SPAN_NAME, EVENT_TYPE 
                FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
                WHERE APPLICATION_NAME = %s
                ORDER BY EVENT_TIMESTAMP DESC 
                LIMIT 5
            """, params=[app_name]).collect()
            
            print("\n   Recent events:")
            for event in recent_events:
                print(f"     {event['EVENT_TIMESTAMP']} - {event['SPAN_NAME']} ({event['EVENT_TYPE']})")
        else:
            print("ℹ️  No events found yet - data might still be processing")
            
    except Exception as e:
        print(f"❌ Could not access observability data: {e}")
        print("   This might indicate insufficient privileges or setup issues")
    
    # Step 8: Test direct query to see what's available
    print("\n8. Checking what observability data is available...")
    try:
        # Check if we can see any events at all
        all_events = session.sql("""
            SELECT COUNT(*) as total_events
            FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS
        """).collect()
        
        total = all_events[0]['TOTAL_EVENTS']
        print(f"   Total events in observability table: {total}")
        
        if total > 0:
            # Show applications that have events
            apps = session.sql("""
                SELECT APPLICATION_NAME, COUNT(*) as event_count
                FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS
                GROUP BY APPLICATION_NAME
                ORDER BY event_count DESC
                LIMIT 10
            """).collect()
            
            print("   Applications with observability data:")
            for app in apps:
                print(f"     {app['APPLICATION_NAME']}: {app['EVENT_COUNT']} events")
                
    except Exception as e:
        print(f"   Could not query observability table: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Minimal AI Observability test completed!")
    print("\nWhat was tested:")
    print("✓ Snowflake session creation")
    print("✓ Basic Cortex AI functionality")
    print("✓ TruLens instrumentation setup")
    print("✓ Instrumented AI calls with automatic tracing")
    print("✓ Observability data capture verification")
    print("\nNext steps:")
    print("1. Check Snowsight UI: AI & ML → Observability")
    print("2. Look for your application:", app_name)
    print("3. Query SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS directly")
    print("4. If you need evaluations, create datasets later with proper privileges")
    
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
    print("\n" + "=" * 50)
    
    main()
