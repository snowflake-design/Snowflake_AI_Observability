

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

class SimpleRAG:
    """
    Simple RAG application for testing AI Observability
    """
    
    def __init__(self, session: Session):
        self.session = session
    
    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        }
    )
    def retrieve_context(self, query: str) -> list:
        """
        Simple context retrieval - for demo purposes
        In a real scenario, this would query a vector database or search service
        """
        # Mock context based on query type
        if "snowflake" in query.lower():
            context = [
                "Snowflake is a cloud-based data platform that enables data warehousing, data lakes, and data engineering.",
                "Snowflake provides AI and ML capabilities through Cortex services.",
                "Snowflake AI Observability helps track and evaluate AI applications."
            ]
        elif "ai" in query.lower() or "artificial intelligence" in query.lower():
            context = [
                "Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence.",
                "AI includes machine learning, natural language processing, and computer vision.",
                "AI observability helps monitor and evaluate AI system performance."
            ]
        else:
            context = [
                "This is a general context about the query.",
                "Context retrieval is an important part of RAG systems.",
                "Good context leads to better AI responses."
            ]
        
        return context
    
    @instrument(
        span_type=SpanAttributes.SpanType.GENERATION,
        attributes={
            SpanAttributes.GENERATION.INPUT: "prompt",
            SpanAttributes.GENERATION.OUTPUT: "return",
        }
    )
    def generate_completion(self, query: str, context_list: list) -> str:
        """
        Generate answer using Snowflake Cortex Complete
        """
        context_str = "\n".join(context_list)
        
        prompt = f"""
        You are a helpful assistant. Answer the question based on the provided context.
        Be concise and accurate. If the context doesn't contain relevant information, say so.
        
        Context:
        {context_str}
        
        Question: {query}
        
        Answer:
        """
        
        try:
            # Use Snowflake Cortex Complete function
            response = complete("llama3.1-70b", prompt)
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
        """
        Main query method that combines retrieval and generation
        """
        context = self.retrieve_context(query)
        return self.generate_completion(query, context)

def setup_snowflake_environment(session: Session) -> None:
    """
    Set up the Snowflake environment for AI Observability
    """
    print("Setting up Snowflake environment...")
    
    # Create observability database and schema
    session.sql("CREATE DATABASE IF NOT EXISTS ai_observability_test_db").collect()
    session.sql("CREATE SCHEMA IF NOT EXISTS ai_observability_test_db.test_schema").collect()
    session.sql("USE DATABASE ai_observability_test_db").collect()
    session.sql("USE SCHEMA test_schema").collect()
    
    print("✅ Environment setup complete")

def create_test_dataset(session: Session) -> None:
    """
    Create a simple test dataset for evaluation
    """
    print("Creating test dataset...")
    
    # Create test data
    test_data = [
        ("What is Snowflake?", "Snowflake is a cloud-based data platform for data warehousing and analytics."),
        ("What is artificial intelligence?", "AI is the simulation of human intelligence in machines."),
        ("How does machine learning work?", "Machine learning uses algorithms to learn patterns from data."),
        ("What is a data warehouse?", "A data warehouse is a central repository for storing and analyzing data."),
        ("Explain cloud computing", "Cloud computing delivers computing services over the internet.")
    ]
    
    # Create DataFrame
    df = session.create_dataframe(
        test_data,
        schema=["query", "ground_truth_response"]
    )
    
    # Save as table
    df.write.mode("overwrite").save_as_table("TEST_DATASET")
    
    print("✅ Test dataset created: TEST_DATASET")

def main():
    """
    Main function to test AI Observability setup
    """
    print("🚀 Testing Snowflake AI Observability Setup...")
    print("=" * 50)
    
    # Step 1: Create Snowflake session
    print("\n1. Creating Snowflake session...")
    try:
        from snowflake.snowpark import Session
        
        session = Session.builder.configs(SNOWFLAKE_CONFIG).create()
        print("✅ Snowflake session created successfully")
        print(f"   Current role: {session.get_current_role()}")
        print(f"   Current database: {session.get_current_database()}")
        print(f"   Current schema: {session.get_current_schema()}")
        
    except Exception as e:
        print(f"❌ Failed to create Snowflake session: {e}")
        return
    
    # Step 2: Setup environment
    print("\n2. Setting up environment...")
    try:
        setup_snowflake_environment(session)
    except Exception as e:
        print(f"❌ Failed to setup environment: {e}")
        return
    
    # Step 3: Create test dataset
    print("\n3. Creating test dataset...")
    try:
        create_test_dataset(session)
    except Exception as e:
        print(f"❌ Failed to create test dataset: {e}")
        return
    
    # Step 4: Create RAG application
    print("\n4. Creating RAG application...")
    try:
        rag = SimpleRAG(session)
        print("✅ RAG application created")
    except Exception as e:
        print(f"❌ Failed to create RAG application: {e}")
        return
    
    # Step 5: Test single query
    print("\n5. Testing single query...")
    try:
        test_query = "What is Snowflake?"
        response = rag.query(test_query)
        print("✅ Single query test successful")
        print(f"   Query: {test_query}")
        print(f"   Response: {response[:100]}...")
    except Exception as e:
        print(f"❌ Single query test failed: {e}")
        return
    
    # Step 6: Setup TruLens connector
    print("\n6. Setting up TruLens connector...")
    try:
        tru_snowflake_connector = SnowflakeConnector(snowpark_session=session)
        print("✅ TruLens Snowflake connector created")
    except Exception as e:
        print(f"❌ Failed to create TruLens connector: {e}")
        return
    
    # Step 7: Create TruApp
    print("\n7. Creating TruApp...")
    try:
        app_name = "test_rag_app"
        app_version = "v1.0"
        
        tru_rag = TruApp(
            rag,
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
    
    # Step 8: Create and run evaluation
    print("\n8. Creating evaluation run...")
    try:
        run_config = RunConfig(
            run_name="test_run_1",
            dataset_name="TEST_DATASET",
            description="Test run for AI Observability setup",
            label="test_evaluation",
            source_type="TABLE",
            dataset_spec={
                "input": "QUERY",
                "ground_truth_output": "GROUND_TRUTH_RESPONSE",
            },
        )
        
        run: Run = tru_rag.add_run(run_config=run_config)
        print("✅ Evaluation run created")
        print(f"   Run name: {run_config.run_name}")
        
    except Exception as e:
        print(f"❌ Failed to create evaluation run: {e}")
        return
    
    # Step 9: Start the run
    print("\n9. Starting evaluation run...")
    try:
        run.start()
        print("✅ Evaluation run started successfully")
    except Exception as e:
        print(f"❌ Failed to start evaluation run: {e}")
        return
    
    # Step 10: Check observability data
    print("\n10. Checking AI Observability data...")
    try:
        # Check if the events table exists and has data
        result = session.sql("SELECT COUNT(*) FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS").collect()
        count = result[0][0]
        print(f"✅ AI Observability Events table accessible")
        print(f"   Records in table: {count}")
        
        if count > 0:
            print("✅ Observability data is being generated!")
        else:
            print("ℹ️  No records yet - this is normal for first run")
            
    except Exception as e:
        print(f"❌ Could not access observability data: {e}")
        print("   The table might not be created yet")
    
    # Step 11: Compute metrics (optional)
    print("\n11. Computing evaluation metrics...")
    try:
        run.compute_metrics([
            "answer_relevance",
            "context_relevance",
            "groundedness",
        ])
        print("✅ Metrics computation initiated")
    except Exception as e:
        print(f"❌ Failed to compute metrics: {e}")
        print("   This might be normal if the run is still processing")
    
    print("\n" + "=" * 50)
    print("🎉 AI Observability test completed!")
    print("\nNext steps:")
    print("1. Check Snowsight UI: AI & ML → Evaluations")
    print("2. Look for your application:", app_name)
    print("3. Review traces and evaluation results")
    print("4. Query SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS for raw data")
    
    # Cleanup
    session.close()

if __name__ == "__main__":
    # Required packages to install:
    print("Required packages:")
    print("- snowflake-snowpark-python")
    print("- trulens-core")
    print("- trulens-providers-cortex")
    print("- trulens-connectors-snowflake")
    print("- pandas")
    print("\nInstall with:")
    print("pip install snowflake-snowpark-python trulens-core trulens-providers-cortex trulens-connectors-snowflake pandas")
    print("\n" + "=" * 50)
    
    main()
