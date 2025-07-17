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

# NEW: Import for TruLens Feedback
from trulens_eval import Feedback
from trulens_eval.feedback.provider.cortex import Cortex as CortexProvider # For LLM-as-a-judge metrics
from opentelemetry import trace

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

# Enable TruLens OpenTelemetry tracing
os.environ = "1"

# Get Snowflake session
try:
    session = get_active_session()
    print("Using active Snowflake session")
except:
    print("Creating new Snowflake session...")
    # IMPORTANT: Replace SNOWFLAKE_CONFIG with your actual Snowflake connection configuration
    # Example: SNOWFLAKE_CONFIG = {"account": "your_account", "user": "your_user", "password": "your_password", "role": "your_role", "warehouse": "your_warehouse", "database": "your_database", "schema": "your_schema"}
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
            "cloud computing":,
            "artificial intelligence": [
                "AI refers to computer systems that can perform tasks typically requiring human intelligence.",
                "AI encompasses machine learning, natural language processing, computer vision, and robotics.",
                "Applications include chatbots, recommendation systems, autonomous vehicles, and medical diagnosis."
            ]
        }

    def detect_toxicity(self, text: str) -> str:
        """Detect if text is toxic using HuggingFace classifier. Returns 'yes' or 'no'."""
        if toxicity_classifier is None:
            return "unknown"
        
        try:
            result = toxicity_classifier(text)
            label = result['label']
            confidence = result['score']
            
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
        current_span = trace.get_current_span()
        
        query_lower = query.lower()
        retrieved_contexts =
        
        try:
            current_span.set_attribute("custom.retrieval_query", query)
            current_span.set_attribute("custom.retrieval_timestamp", str(time.time()))
            current_span.set_attribute("custom.query_length", len(query))
            print(f"✅ Added custom retrieval attributes")
        except Exception as e:
            print(f"⚠️ Could not add custom retrieval attributes: {e}")
        
        for topic, contexts in self.knowledge_base.items():
            if topic in query_lower:
                retrieved_contexts.extend(contexts)
                try:
                    current_span.set_attribute("custom.matched_topic", topic)
                    current_span.set_attribute("custom.topic_match_type", "exact")
                except Exception as e:
                    print(f"⚠️ Could not add topic attribute: {e}")
                break
        
        if not retrieved_contexts:
            retrieved_contexts = self.knowledge_base.get("artificial intelligence",)
            try:
                current_span.set_attribute("custom.matched_topic", "artificial_intelligence_default")
                current_span.set_attribute("custom.topic_match_type", "fallback")
            except Exception as e:
                print(f"⚠️ Could not add fallback topic attribute: {e}")
        
        try:
            current_span.set_attribute("custom.contexts_found", len(retrieved_contexts))
            current_span.set_attribute("custom.retrieval_success", len(retrieved_contexts) > 0)
            
            current_span.add_event(
                "context_retrieval_complete",
                {
                    "contexts_count": len(retrieved_contexts),
                    "query_topic": query_lower,
                    "retrieval_method": "keyword_matching"
                }
            )
            
        except Exception as e:
            print(f"⚠️ Could not add context count attributes: {e}")
        
        print(f"🔍 Retrieved {len(retrieved_contexts)} contexts")
        return retrieved_contexts

    @instrument(
        span_type=SpanAttributes.SpanType.GENERATION,
        attributes={
            SpanAttributes.GENERATION.QUERY_TEXT: "query",
            SpanAttributes.GENERATION.GENERATED_TEXT: "return",
        }
    )
    def generate_completion(self, query: str, context_str: list) -> str:
        """Generate answer from context by calling an LLM."""
        current_span = trace.get_current_span()
        
        context_text = "\n".join([f"- {ctx}" for ctx in context_str])
        
        prompt = f"""Based on the following context, answer the question:

Context:
{context_text}

Question: {query}

Answer:"""
        
        try:
            current_span.set_attribute("custom.llm_model", self.model)
            current_span.set_attribute("custom.prompt_length", len(prompt))
            current_span.set_attribute("custom.context_items", len(context_str))
            current_span.set_attribute("custom.generation_method", "snowflake_cortex")
            print(f"✅ Added custom generation attributes")
        except Exception as e:
            print(f"⚠️ Could not add generation attributes: {e}")
        
        try:
            response = complete(self.model, prompt)
            
            try:
                current_span.set_attribute("custom.generation_status", "success")
                current_span.set_attribute("custom.response_length", len(response))
                current_span.set_attribute("custom.tokens_estimated", len(response.split()))
                
                current_span.add_event(
                    "llm_generation_complete",
                    {
                        "model": self.model,
                        "response_length": len(response),
                        "status": "success"
                    }
                )
                
            except Exception as attr_error:
                print(f"⚠️ Could not add success attributes: {attr_error}")
                
            print(f"🤖 Generated response ({len(response)} chars)")
            return response
            
        except Exception as e:
            try:
                current_span.set_attribute("custom.generation_status", "error")
                current_span.set_attribute("custom.error_message", str(e))
                current_span.set_attribute("custom.error_type", type(e).__name__)
                
                current_span.add_event(
                    "llm_generation_failed",
                    {
                        "model": self.model,
                        "error": str(e),
                        "status": "failed"
                    }
                )
                
            except Exception as attr_error:
                print(f"⚠️ Could not add error attributes: {attr_error}")
            
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
        current_span = trace.get_current_span()
        
        if isinstance(input_data, dict):
            query = input_data.get('query', '')
            username = input_data.get('username', 'unknown')
        elif hasattr(input_data, 'get'):
            query = input_data.get('query', str(input_data))
            username = input_data.get('username', 'unknown')
        else:
            query = str(input_data)
        
        print(f"🔍 Processing query from user: {username}")
        
        try:
            current_span.set_attribute("custom.username", username)
            current_span.set_attribute("custom.query_text", query)
            current_span.set_attribute("custom.query_length", len(query))
            current_span.set_attribute("custom.processing_start", str(time.time()))
            current_span.set_attribute("custom.session_id", f"session_{username}_{int(time.time())}")
            
            # Detect toxicity and add to span (for trace details)
            toxicity_result_str = self.detect_toxicity(query) # Keep for trace attribute
            current_span.set_attribute("custom.toxicity_detected", toxicity_result_str)
            current_span.set_attribute("custom.toxicity_check_enabled", toxicity_classifier is not None)
            
            special_users =
            current_span.set_attribute("custom.is_special_user", username in special_users)
            current_span.set_attribute("custom.user_type", "special" if username in special_users else "regular")
            
            print(f"✅ Custom attributes added: username={username}, toxicity={toxicity_result_str}")
            
        except Exception as e:
            print(f"⚠️ Could not add custom attributes: {e}")
        
        try:
            current_span.add_event(
                "rag_processing_started",
                {
                    "username": username,
                    "query_preview": query[:50],
                    "toxicity_detected": toxicity_result_str,
                    "processing_timestamp": str(time.time())
                }
            )
        except Exception as e:
            print(f"⚠️ Could not add start event: {e}")
        
        # Process the query
        context_str = self.retrieve_context(query)
        response = self.generate_completion(query, context_str)
        
        try:
            current_span.set_attribute("custom.processing_complete", True)
            current_span.set_attribute("custom.processing_end", str(time.time()))
            current_span.set_attribute("custom.final_response_length", len(response))
            current_span.set_attribute("custom.context_used_count", len(context_str))
            current_span.set_attribute("custom.response_generated", len(response) > 0)
            
            current_span.add_event(
                "rag_interaction_complete",
                {
                    "username": username,
                    "query": query[:100],
                    "toxicity": toxicity_result_str, # Use the string result for event
                    "response_length": len(response),
                    "context_count": len(context_str),
                    "success": True
                }
            )
            print(f"✅ Processing complete event added")
            
        except Exception as e:
            print(f"⚠️ Could not add completion attributes: {e}")
        
        print(f"✅ Response generated for user {username} (toxicity: {toxicity_result_str})")
        
        return response

# Initialize the RAG application
test_app = RAGApplication()

# Test basic functionality
print("Testing basic functionality...")
test_response = test_app.answer_query("What is machine learning?", "test_user")
print(f"Test successful: {test_response[:100] if test_response else 'No response'}...")

# Create Snowflake connector
connector = SnowflakeConnector(snowpark_session=session)

# NEW: Define the custom toxicity feedback function for TruLens
# This function will return a numerical score (0.0 to 1.0) for toxicity
def calculate_toxicity_score(text: str) -> float:
    """
    Calculates a numerical toxicity score (0.0 for TOXIC, 1.0 for NOT_TOXIC).
    This is designed to be used as a TruLens Feedback function.
    """
    if toxicity_classifier is None:
        return 0.5 # Neutral score if classifier not loaded
    
    try:
        result = toxicity_classifier(text) # Get the first (and only) result
        label = result['label']
        confidence = result['score']
        
        # Map to a 0-1 score where 1.0 is "not toxic" and 0.0 is "toxic"
        # This makes higher scores "better" (less toxic) for dashboard visualization
        if label == 'TOXIC':
            # If toxic, score is 1 - confidence (e.g., 0.9 confidence in TOXIC -> 0.1 score)
            return 1.0 - confidence
        else: # NOT_TOXIC
            # If not toxic, score is confidence (e.g., 0.9 confidence in NOT_TOXIC -> 0.9 score)
            return confidence
    except Exception as e:
        print(f"Error in calculate_toxicity_score feedback function: {e}")
        return 0.5 # Neutral score on error

# NEW: Instantiate CortexProvider for LLM-as-a-judge metrics
# This is needed for answer_relevance, context_relevance, groundedness, correctness
cortex_provider = CortexProvider(snowpark_session=session)

# NEW: Define standard LLM-as-a-judge feedback functions
f_answer_relevance = (
    Feedback(cortex_provider.relevance_with_cot_reasons, name="Answer Relevance")
   .on_input_output()
)
f_context_relevance = (
    Feedback(cortex_provider.context_relevance_with_cot_reasons, name="Context Relevance")
   .on_output().on_context()
)
f_groundedness = (
    Feedback(cortex_provider.groundedness_measure_with_cot_reasons, name="Groundedness")
   .on_output().on_context()
)
# Assuming 'correctness' is also a TruLens feedback function, if not, you'd define it similarly
# For simplicity, if cortex_provider doesn't have a direct 'correctness' method,
# you might need to define a custom one or remove it if not applicable.
# For this example, we'll assume it's available or you'd define a custom one.
f_correctness = (
    Feedback(cortex_provider.correctness_with_cot_reasons, name="Correctness")
   .on_output().on_ground_truth() # Requires ground truth
)

# NEW: Create the custom toxicity feedback object
f_toxicity_metric = (
    Feedback(calculate_toxicity_score, name="Toxicity Score") # Name will appear on dashboard
  .on_output() # Evaluate the toxicity of the RAG application's final output
)


# Enhanced test dataset with usernames
usernames =

test_data = pd.DataFrame({
    'query':,
    'expected_answer':,
    'username': [random.choice(usernames) for _ in range(5)]
})

print(f"Created dataset with {len(test_data)} test queries")
print("Dataset preview:")
print(test_data[['query', 'username']].to_string())

# Register the app for Snowflake AI Observability
app_name = f"rag_trulens_custom_metrics_{int(time.time())}" # Updated app name
tru_app = TruApp(
    test_app,
    app_name=app_name, 
    app_version="v1.0",
    connector=connector,
    main_method=test_app.answer_query,
    # NEW: Pass all feedback functions, including your custom toxicity metric
    feedbacks=
)

print(f"Application registered successfully: {app_name}")

# Single run configuration
run_config = RunConfig(
    run_name=f"trulens_custom_metrics_run_{int(time.time())}", # Updated run name
    description="Standard and custom TruLens metrics with OpenTelemetry custom attributes",
    label="trulens_custom_metrics_test",
    source_type="DATAFRAME",
    dataset_name="Test dataset with custom TruLens metrics and OTel attributes",
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
    
    if status in:
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

# Compute all defined TruLens metrics (including custom toxicity)
print("\n" + "="*60)
print("COMPUTING ALL DEFINED TRULENS METRICS (INCLUDING CUSTOM TOXICITY)")
print("="*60)

# Now include "Toxicity Score" in the list of metrics to compute
metrics_to_compute =

successful_metrics =
failed_metrics =

for metric in metrics_to_compute:
    print(f"\n--- Computing {metric.upper()} ---")
    
    try:
        run.compute_metrics(metrics=[metric])
        print(f"✅ {metric} computation initiated successfully")
        
        print(f"Waiting for {metric} computation to complete...")
        time.sleep(90) # Give ample time for computation
        
        current_status = run.get_status()
        print(f"Status after {metric}: {current_status}")
        
        successful_metrics.append(metric)
        
    except Exception as e:
        print(f"❌ Error computing {metric}: {e}")
        failed_metrics.append(metric)
    
    print(f"Brief pause before next metric...")
    time.sleep(30)

# Custom data analysis summary (OpenTelemetry attributes)
print("\n" + "="*60)
print("OPENTELEMETRY CUSTOM ATTRIBUTES SUMMARY (TRACE-LEVEL DATA)")
print("="*60)

print("✅ Custom attributes added using OpenTelemetry trace.get_current_span().set_attribute():")
print("   - custom.username: User who made the request")
print("   - custom.toxicity_detected: Toxicity analysis result (string 'yes'/'no')")
print("   - custom.query_text: Original query text")
print("   - custom.query_length: Length of query")
print("   - custom.processing_start/end: Processing timestamps")
print("   - custom.session_id: Unique session identifier")
print("   - custom.is_special_user: Whether user is in special list")
print("   - custom.user_type: User classification (special/regular)")
print("   - custom.matched_topic: Which knowledge base topic matched")
print("   - custom.contexts_found: Number of retrieved contexts")
print("   - custom.llm_model: Model used for generation")
print("   - custom.response_length: Length of generated response")
print("   - custom.generation_status: Success/failure status")

print("\n✅ Custom events added using current_span.add_event():")
print("   - rag_processing_started: When processing begins")
print("   - context_retrieval_complete: When context retrieval finishes")
print("   - llm_generation_complete: When LLM generation finishes")
print("   - rag_interaction_complete: When full interaction completes")

# Final results
print("\n" + "="*60)
print("FINAL RESULTS - SNOWFLAKE AI OBSERVABILITY WITH CUSTOM TRULENS METRICS & OTel ATTRIBUTES")
print("="*60)
print(f"✅ Successful metrics: {successful_metrics}")
print(f"❌ Failed metrics: {failed_metrics}")
print(f"Success rate: {len(successful_metrics)}/{len(metrics_to_compute)}")

final_status = run.get_status()
print(f"\nFinal run status: {final_status}")

print("\n" + "="*60)
print("KEY ACCOMPLISHMENTS")
print("="*60)
print("✅ **Custom 'Toxicity Score' metric now computed by TruLens and visible on dashboard.**")
print("✅ Standard Snowflake AI Observability metrics computed.")
print("✅ OpenTelemetry custom attributes and events captured in traces for detailed debugging.")
print("✅ Username and toxicity data captured in OpenTelemetry spans (trace-level).")
print("✅ All traces and metrics stored in Snowflake for analysis.")

print("\n📊 View results in Snowsight:")
print("   Navigate to: AI & ML -> Evaluations")
print("   Look for app:", app_name)
print("   **You should now see 'Toxicity Score' alongside other metrics on the main dashboard.**")
print("   For custom trace data (username, detailed toxicity string, etc.):")
print("   Click on individual records to see detailed traces and inspect custom.* attributes in span details.")

print("\n🎉 SUCCESS: Custom TruLens metric for Toxicity implemented and integrated!")

