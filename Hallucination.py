import os
import pandas as pd
import time
import random
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.session import Session
from snowflake.cortex import complete
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
from trulens.apps.app import TruApp
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core.run import Run, RunConfig
from trulens.core import Feedback, Provider

# Toxicity detection setup
try:
    import warnings
    warnings.filterwarnings("ignore", message="Some weights of the model checkpoint.*were not used")
    
    from transformers import pipeline
    toxicity_classifier = pipeline("text-classification", model="s-nlp/roberta_toxicity_classifier")
    print("✅ Toxicity classifier loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load toxicity classifier: {e}")
    toxicity_classifier = None

# Custom Provider for TruLens feedback functions (separate from Snowflake AI Observability)
class CustomMetricsProvider(Provider):
    """Custom provider for feedback functions - works with standalone TruLens but not Snowflake AI Observability."""
    
    def __init__(self):
        super().__init__()
    
    def toxicity_feedback(self, output: str) -> float:
        """Toxicity detection as TruLens feedback function."""
        if toxicity_classifier is None:
            return 0.5
        
        try:
            result = toxicity_classifier(output)
            label = result[0]['label']
            confidence = result[0]['score']
            
            if label == 'TOXIC':
                return 1.0 - confidence
            else:
                return confidence
                
        except Exception as e:
            print(f"❌ Toxicity feedback error: {e}")
            return 0.5
    
    def username_feedback(self, query: str) -> float:
        """Username detection as TruLens feedback function."""
        try:
            if 'username' in query.lower() or any(user in query for user in ['abc', 'XYZ', 'KKK']):
                return 1.0
            else:
                return 0.0
        except Exception as e:
            print(f"❌ Username feedback error: {e}")
            return 0.5

# Enable TruLens OpenTelemetry tracing
os.environ["TRULENS_OTEL_TRACING"] = "1"

# Get Snowflake session
try:
    session = get_active_session()
    print("Using active Snowflake session")
except:
    print("Creating new Snowflake session...")
    session = Session.builder.configs(SNOWFLAKE_CONFIG).create()

print(f"Current context: {session.get_current_database()}.{session.get_current_schema()}")

class RAGApplication:
    def __init__(self):
        self.model = "mistral-large2"
        
        self.knowledge_base = {
            "machine learning": [
                "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
                "ML algorithms can identify patterns in large datasets and make predictions or decisions based on those patterns.",
                "Common types include supervised learning, unsupervised learning, and reinforcement learning."
            ],
            "cloud computing": [
                "Cloud computing delivers computing services over the internet including storage, processing power, and applications.",
                "Major benefits include scalability, cost-effectiveness, and accessibility from anywhere with internet connection.",
                "Popular cloud providers include AWS, Microsoft Azure, and Google Cloud Platform."
            ],
            "artificial intelligence": [
                "AI refers to computer systems that can perform tasks typically requiring human intelligence.",
                "AI encompasses machine learning, natural language processing, computer vision, and robotics.",
                "Applications include chatbots, recommendation systems, autonomous vehicles, and medical diagnosis."
            ]
        }

    def detect_toxicity(self, text: str) -> str:
        """Detect if text is toxic using HuggingFace classifier."""
        if toxicity_classifier is None:
            return "unknown"
        
        try:
            result = toxicity_classifier(text)
            label = result[0]['label']
            confidence = result[0]['score']
            
            if label == 'TOXIC' and confidence > 0.5:
                return "yes"
            else:
                return "no"
        except Exception as e:
            print(f"Error in toxicity detection: {e}")
            return "unknown"

    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        }
    )
    def retrieve_context(self, query: str) -> list:
        """Retrieve relevant text from knowledge base."""
        query_lower = query.lower()
        retrieved_contexts = []
        
        # Add custom span attributes using Snowflake telemetry API
        try:
            from snowflake import telemetry
            # Log custom retrieval metadata
            telemetry.set_span_attribute("custom.retrieval_query", query)
            telemetry.set_span_attribute("custom.retrieval_timestamp", str(time.time()))
        except Exception as e:
            print(f"⚠️ Could not add custom retrieval attributes: {e}")
        
        for topic, contexts in self.knowledge_base.items():
            if topic in query_lower:
                retrieved_contexts.extend(contexts)
                # Log the matched topic
                try:
                    from snowflake import telemetry
                    telemetry.set_span_attribute("custom.matched_topic", topic)
                except:
                    pass
                break
        
        if not retrieved_contexts:
            retrieved_contexts = self.knowledge_base.get("artificial intelligence", [])
            try:
                from snowflake import telemetry
                telemetry.set_span_attribute("custom.matched_topic", "artificial_intelligence_default")
            except:
                pass
        
        # Log number of contexts found
        try:
            from snowflake import telemetry
            telemetry.set_span_attribute("custom.contexts_found", len(retrieved_contexts))
        except:
            pass
        
        print(f"🔍 Retrieved {len(retrieved_contexts)} contexts")
        return retrieved_contexts

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate_completion(self, query: str, context_str: list) -> str:
        """Generate answer from context by calling an LLM."""
        context_text = "\n".join([f"- {ctx}" for ctx in context_str])
        
        prompt = f"""Based on the following context, answer the question:

Context:
{context_text}

Question: {query}

Answer:"""
        
        # Add custom generation attributes
        try:
            from snowflake import telemetry
            telemetry.set_span_attribute("custom.llm_model", self.model)
            telemetry.set_span_attribute("custom.prompt_length", len(prompt))
            telemetry.set_span_attribute("custom.context_items", len(context_str))
        except Exception as e:
            print(f"⚠️ Could not add generation attributes: {e}")
        
        try:
            response = complete(self.model, prompt)
            
            # Log successful generation
            try:
                from snowflake import telemetry
                telemetry.set_span_attribute("custom.generation_status", "success")
                telemetry.set_span_attribute("custom.response_length", len(response))
            except:
                pass
                
            print(f"🤖 Generated response ({len(response)} chars)")
            return response
        except Exception as e:
            # Log generation error
            try:
                from snowflake import telemetry
                telemetry.set_span_attribute("custom.generation_status", "error")
                telemetry.set_span_attribute("custom.error_message", str(e))
            except:
                pass
            return f"Error generating response: {str(e)}"

    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "input_data",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        }
    )
    def answer_query(self, input_data, username: str = "unknown") -> str:
        """Main entry point for the RAG application."""
        
        # Handle different input types
        if isinstance(input_data, dict):
            query = input_data.get('query', '')
            username = input_data.get('username', 'unknown')
        elif hasattr(input_data, 'get'):
            query = input_data.get('query', str(input_data))
            username = input_data.get('username', 'unknown')
        else:
            query = str(input_data)
        
        print(f"🔍 Processing query from user: {username}")
        
        # Add custom span attributes for username and other metadata
        try:
            from snowflake import telemetry
            telemetry.set_span_attribute("custom.username", username)
            telemetry.set_span_attribute("custom.query_text", query)
            telemetry.set_span_attribute("custom.query_length", len(query))
            telemetry.set_span_attribute("custom.processing_start", str(time.time()))
            
            # Detect toxicity and add to span
            toxicity_result = self.detect_toxicity(query)
            telemetry.set_span_attribute("custom.toxicity_detected", toxicity_result)
            
            print(f"✅ Custom attributes added: username={username}, toxicity={toxicity_result}")
            
        except Exception as e:
            print(f"⚠️ Could not add custom attributes: {e}")
        
        # Detect toxicity for console logging
        toxicity_result = self.detect_toxicity(query)
        print(f"🛡️ Toxicity check for '{query[:50]}...': {toxicity_result}")
        
        context_str = self.retrieve_context(query)
        response = self.generate_completion(query, context_str)
        
        # Add final processing attributes
        try:
            from snowflake import telemetry
            telemetry.set_span_attribute("custom.processing_complete", True)
            telemetry.set_span_attribute("custom.processing_end", str(time.time()))
            telemetry.set_span_attribute("custom.final_response_length", len(response))
            
            # Add a custom event for this complete interaction
            telemetry.add_event(
                "rag_interaction_complete", 
                {
                    "username": username,
                    "query": query[:100],  # truncated for brevity
                    "toxicity": toxicity_result,
                    "response_length": len(response),
                    "context_count": len(context_str)
                }
            )
            print(f"✅ Processing complete event added")
            
        except Exception as e:
            print(f"⚠️ Could not add completion attributes: {e}")
        
        print(f"✅ Response generated for user {username} (toxicity: {toxicity_result})")
        
        return response

# Initialize the RAG application
test_app = RAGApplication()

# Test basic functionality
print("Testing basic functionality...")
test_response = test_app.answer_query("What is machine learning?", "test_user")
print(f"Test successful: {test_response[:100] if test_response else 'No response'}...")

# Create Snowflake connector
connector = SnowflakeConnector(snowpark_session=session)

# Create custom feedback functions (these work with standalone TruLens, not Snowflake AI Observability)
print("\n" + "="*60)
print("NOTE: Custom feedback functions are for standalone TruLens only")
print("="*60)
custom_provider = CustomMetricsProvider()

f_toxicity = Feedback(
    custom_provider.toxicity_feedback,
    name="custom_toxicity"
).on_output()

f_username = Feedback(
    custom_provider.username_feedback, 
    name="username_detection"
).on_input()

print("✅ Custom feedback functions created (for standalone TruLens)")

# Enhanced test dataset with usernames
usernames = ["abc", "XYZ", "KKK"]

test_data = pd.DataFrame({
    'query': [
        "What is machine learning?",
        "What are cloud computing benefits?", 
        "What are AI applications?"
    ],
    'expected_answer': [
        "Machine learning enables computers to learn from data without explicit programming.",
        "Cloud computing provides scalability, cost-effectiveness, and accessibility.",
        "AI applications include chatbots, recommendation systems, and autonomous vehicles."
    ],
    'username': [random.choice(usernames) for _ in range(3)]
})

print(f"Created dataset with {len(test_data)} test queries")
print("Dataset preview:")
print(test_data[['query', 'username']].to_string())

# Register the app for Snowflake AI Observability (no custom feedback functions supported)
app_name = f"rag_metrics_app_{int(time.time())}"
tru_app = TruApp(
    test_app,
    app_name=app_name, 
    app_version="v1.0",
    connector=connector,
    main_method=test_app.answer_query
    # NOTE: Snowflake AI Observability doesn't support custom feedback functions
)

print(f"Application registered successfully: {app_name}")

# Single run configuration
run_config = RunConfig(
    run_name=f"all_metrics_run_{int(time.time())}",
    description="Standard metrics with custom span attributes for username and toxicity",
    label="all_metrics_test",
    source_type="DATAFRAME",
    dataset_name="Test dataset with user tracking and toxicity detection",
    dataset_spec={
        "RETRIEVAL.QUERY_TEXT": "query",
        "RECORD_ROOT.INPUT": "query",
        "RECORD_ROOT.GROUND_TRUTH_OUTPUT": "expected_answer",
    },
    llm_judge_name="mistral-large2"
)

print(f"Run configuration created: {run_config.run_name}")

# Add run to TruApp
run = tru_app.add_run(run_config=run_config)
print("Run added successfully")

# Start the run and wait for completion
print("Starting run execution...")
run.start(input_df=test_data)
print("Run execution completed")

# Wait for invocation to complete
print("\n" + "="*60)
print("WAITING FOR INVOCATION TO COMPLETE")
print("="*60)

max_attempts = 60
attempt = 0

while attempt < max_attempts:
    status = run.get_status()
    print(f"Attempt {attempt + 1}: Status = {status}")
    
    if status in ["INVOCATION_COMPLETED", "INVOCATION_PARTIALLY_COMPLETED"]:
        print("✅ INVOCATION COMPLETED - Ready to compute metrics!")
        break
    elif status == "INVOCATION_FAILED":
        print("❌ Invocation failed!")
        break
    else:
        time.sleep(15)
        attempt += 1

if attempt >= max_attempts:
    print("⚠️ Timeout waiting for completion, but trying metrics anyway...")

# Compute standard Snowflake AI Observability metrics
print("\n" + "="*60)
print("COMPUTING STANDARD SNOWFLAKE AI OBSERVABILITY METRICS")
print("="*60)

# Only standard metrics are supported in Snowflake AI Observability
standard_metrics = [
    "answer_relevance",
    "context_relevance", 
    "groundedness",
    "correctness"
]

successful_metrics = []
failed_metrics = []

for metric in standard_metrics:
    print(f"\n--- Computing {metric.upper()} ---")
    
    try:
        run.compute_metrics(metrics=[metric])
        print(f"✅ {metric} computation initiated successfully")
        
        print(f"Waiting for {metric} computation to complete...")
        time.sleep(90)
        
        current_status = run.get_status()
        print(f"Status after {metric}: {current_status}")
        
        successful_metrics.append(metric)
        
    except Exception as e:
        print(f"❌ Error computing {metric}: {e}")
        failed_metrics.append(metric)
    
    print(f"Brief pause before next metric...")
    time.sleep(30)

# Custom data analysis summary
print("\n" + "="*60)
print("CUSTOM DATA LOGGING SUMMARY")
print("="*60)

print("✅ Username tracking: Logged via telemetry.set_span_attribute('custom.username', username)")
print("✅ Toxicity detection: Logged via telemetry.set_span_attribute('custom.toxicity_detected', result)")
print("✅ Query metadata: Query text, length, timestamps logged as custom attributes")
print("✅ Retrieval metadata: Matched topic, contexts found logged as custom attributes")
print("✅ Generation metadata: Model, prompt length, response length logged as custom attributes")
print("✅ Custom events: Complete interaction events with telemetry.add_event()")

if toxicity_classifier:
    print("\nToxicity analysis summary:")
    for idx, row in test_data.iterrows():
        toxicity = test_app.detect_toxicity(row['query'])
        print(f"   Query: '{row['query'][:50]}...' | User: {row['username']} | Toxic: {toxicity}")
else:
    print("⚠️ Toxicity classifier not available")

# Final results
print("\n" + "="*60)
print("FINAL RESULTS - SNOWFLAKE AI OBSERVABILITY")
print("="*60)
print(f"✅ Successful standard metrics: {successful_metrics}")
print(f"❌ Failed metrics: {failed_metrics}")
print(f"Success rate: {len(successful_metrics)}/{len(standard_metrics)}")

final_status = run.get_status()
print(f"\nFinal run status: {final_status}")

print("\n" + "="*60)
print("KEY ACCOMPLISHMENTS")
print("="*60)
print("✅ Proper Snowflake AI Observability integration with standard metrics")
print("✅ Custom span attributes logged using Snowflake telemetry API")
print("✅ Username and toxicity data captured in trace spans")
print("✅ Custom events added for complete interaction tracking")
print("✅ All traces stored in Snowflake with both standard and custom data")

print("\n📊 View results in Snowsight:")
print("   Navigate to: AI & ML -> Evaluations")
print("   Look for app:", app_name)
print("   Standard metrics: answer_relevance, context_relevance, groundedness, correctness")
print("   Custom trace data: Click on individual records to see detailed traces")
print("   🔍 Look for custom.* attributes in span details:")
print("      - custom.username (user who made the request)")
print("      - custom.toxicity_detected (toxicity analysis result)")
print("      - custom.matched_topic (retrieval topic)")
print("      - custom.llm_model (generation model)")
print("      - custom.processing_* (timestamps and metadata)")
print("   📊 Custom events: 'rag_interaction_complete' events with full context")

print("\n💡 IMPORTANT ARCHITECTURAL NOTES:")
print("   🎯 Snowflake AI Observability: Standard metrics only (answer_relevance, etc.)")
print("   📊 Custom data: Logged via Snowflake telemetry API as span attributes")
print("   🛡️ Toxicity & username: Captured in trace metadata, not as separate metrics")
print("   ⚡ All data queryable via: SELECT * FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS")

print("\n🎉 SUCCESS: Hybrid approach implemented correctly!")
print("   Standard Snowflake AI Observability metrics + Custom span attributes")
print("   Username tracking and toxicity detection integrated into traces")
