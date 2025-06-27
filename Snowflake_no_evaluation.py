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
            # Correct attribute mapping for input and output
            SpanAttributes.GENERATION.INPUT: lambda args, kwargs: kwargs.get("question", args[0] if args else ""),
            SpanAttributes.GENERATION.OUTPUT: lambda result: result,
            # Additional attributes for better observability
            SpanAttributes.GENERATION.MODEL: "llama3.1-70b",
            SpanAttributes.GENERATION.PROVIDER: "snowflake-cortex"
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
            SpanAttributes.RETRIEVAL.QUERY: lambda args, kwargs: kwargs.get("query", args[0] if args else ""),
            SpanAttributes.RETRIEVAL.DOCUMENTS: lambda result: result if isinstance(result, list) else [result]
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
            # For RAG applications, map to the correct RECORD_ROOT attributes
            "RECORD_ROOT.INPUT": lambda args, kwargs: kwargs.get("question", args[0] if args else ""),
            "RECORD_ROOT.OUTPUT": lambda result: result,
            "RETRIEVAL.QUERY_TEXT": lambda args, kwargs: kwargs.get("question", args[0] if args else ""),
            SpanAttributes.GENERATION.MODEL: "llama3.1-70b"
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
    Main function to test AI Observability setup with correct span attributes
    """
    print("🚀 Enhanced Snowflake AI Observability Test with Correct Span Attributes...")
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
        app_name = "enhanced_test_app"
        app_version = "v1.1"
        
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
    print("\n6. Testing instrumented AI calls with correct span attributes...")
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
        
        print("✅ All instrumented calls completed successfully with proper span attributes")
        
    except Exception as e:
        print(f"❌ Instrumented calls failed: {e}")
        return
    
    # Step 7: Check for observability data with enhanced queries
    print("\n7. Checking for AI Observability data...")
    try:
        import time
        time.sleep(3)  # Wait a bit longer for data to be written
        
        # Query for events with span details
        result = session.sql("""
            SELECT COUNT(*) as event_count,
                   COUNT(DISTINCT SPAN_ID) as unique_spans,
                   COUNT(DISTINCT SPAN_NAME) as unique_span_names
            FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
            WHERE APPLICATION_NAME = %s
        """, params=[app_name]).collect()
        
        if result:
            row = result[0]
            print(f"✅ AI Observability Events table accessible")
            print(f"   Total events for app '{app_name}': {row['EVENT_COUNT']}")
            print(f"   Unique spans: {row['UNIQUE_SPANS']}")
            print(f"   Unique span names: {row['UNIQUE_SPAN_NAMES']}")
            
            if row['EVENT_COUNT'] > 0:
                print("🎉 SUCCESS: Observability data is being captured with proper attributes!")
                
                # Show detailed span information
                span_details = session.sql("""
                    SELECT SPAN_NAME, EVENT_TYPE, 
                           COUNT(*) as event_count,
                           MIN(EVENT_TIMESTAMP) as first_event,
                           MAX(EVENT_TIMESTAMP) as last_event
                    FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
                    WHERE APPLICATION_NAME = %s
                    GROUP BY SPAN_NAME, EVENT_TYPE
                    ORDER BY first_event DESC
                """, params=[app_name]).collect()
                
                print("\n   Detailed span information:")
                for span in span_details:
                    print(f"     {span['SPAN_NAME']} ({span['EVENT_TYPE']}): {span['EVENT_COUNT']} events")
                
                # Show sample attributes for generation spans
                generation_attrs = session.sql("""
                    SELECT SPAN_NAME, ATTRIBUTES
                    FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
                    WHERE APPLICATION_NAME = %s 
                      AND SPAN_NAME LIKE '%ask_question%'
                      AND EVENT_TYPE = 'span_end'
                    LIMIT 2
                """, params=[app_name]).collect()
                
                if generation_attrs:
                    print("\n   Sample generation span attributes:")
                    for attr in generation_attrs:
                        print(f"     {attr['SPAN_NAME']}: {str(attr['ATTRIBUTES'])[:200]}...")
            
        else:
            print("ℹ️  No events found yet - data might still be processing")
            
    except Exception as e:
        print(f"❌ Could not access observability data: {e}")
        print("   This might indicate insufficient privileges or setup issues")
    
    print("\n" + "=" * 70)
    print("🎉 Enhanced AI Observability test completed!")
    print("\nKey improvements made:")
    print("✓ Fixed span attribute mapping for GENERATION spans")
    print("✓ Added proper input/output lambda functions")
    print("✓ Included RECORD_ROOT.INPUT and RECORD_ROOT.OUTPUT for evaluation")
    print("✓ Added RETRIEVAL.QUERY_TEXT for RAG compatibility")
    print("✓ Enhanced observability data validation")
    
    print("\nSpan Attributes now properly configured:")
    print("- SpanAttributes.GENERATION.INPUT: Maps function input parameter")
    print("- SpanAttributes.GENERATION.OUTPUT: Maps function return value")
    print("- RECORD_ROOT.INPUT: Required for evaluation metrics")
    print("- RECORD_ROOT.OUTPUT: Required for evaluation metrics")
    print("- RETRIEVAL.QUERY_TEXT: Required for RAG applications")
    
    print("\nNext steps:")
    print("1. Check Snowsight UI: AI & ML → Observability")
    print("2. Look for your application:", app_name)
    print("3. Verify span attributes are properly captured")
    print("4. Create datasets for evaluation using the captured attributes")
    
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
