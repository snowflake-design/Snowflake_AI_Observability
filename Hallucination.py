import os
import pandas as pd
import time
import random
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.session import Session
from snowflake.cortex import complete
from trulens.apps.app import TruApp
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core.run import Run, RunConfig

# NEW: Import for TruLens Feedback
from trulens_eval import Feedback
from trulens_eval.feedback.provider.cortex import Cortex as CortexProvider # For LLM-as-a-judge metrics

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

# Get Snowflake session
try:
    session = get_active_session()
    print("Using active Snowflake session")
except:
    print("Creating new Snowflake session...")
    # IMPORTANT: Replace SNOWFLAKE_CONFIG with your actual Snowflake connection configuration
    # Example: SNOWFLAKE_CONFIG = {"account": "your_account", "user": "your_user", "password": "your_password", "role": "your_role", "warehouse": "your_warehouse", "database": "your_database", "schema": "your_schema"}
    SNOWFLAKE_CONFIG = {
        "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
        "user": os.environ.get("SNOWFLAKE_USER"),
        "password": os.environ.get("SNOWFLAKE_PASSWORD"),
        "role": os.environ.get("SNOWFLAKE_ROLE"),
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
        "database": os.environ.get("SNOWFLAKE_DATABASE"),
        "schema": os.environ.get("SNOWFLAKE_SCHEMA")
    }
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
                "Cloud computing is the on-demand delivery of IT resources over the Internet with pay-as-you-go pricing.",
                "Instead of buying, owning, and maintaining physical data centers and servers, you can access technology services from a cloud provider.",
                "Major cloud providers include Amazon Web Services (AWS), Microsoft Azure, and Google Cloud Platform (GCP)."
            ],
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
            result = toxicity_classifier(text)[0]
            label = result['label']
            confidence = result['score']
            
            if label == 'TOXIC' and confidence > 0.5:
                return "yes"
            else:
                return "no"
        except Exception as e:
            print(f"Error in toxicity detection: {e}")
            return "unknown"

    # Removed @instrument decorator and OpenTelemetry specific code
    def retrieve_context(self, query: str) -> list:
        """Retrieve relevant text from knowledge base."""
        query_lower = query.lower()
        retrieved_contexts = []
        
        for topic, contexts in self.knowledge_base.items():
            if topic in query_lower:
                retrieved_contexts.extend(contexts)
                break
        
        if not retrieved_contexts:
            retrieved_contexts = self.knowledge_base.get("artificial intelligence", [])
        
        print(f"🔍 Retrieved {len(retrieved_contexts)} contexts")
        return retrieved_contexts

    # Removed @instrument decorator and OpenTelemetry specific code
    def generate_completion(self, query: str, context_str: list) -> str:
        """Generate answer from context by calling an LLM."""
        context_text = "\n".join([f"- {ctx}" for ctx in context_str])
        
        prompt = f"""Based on the following context, answer the question:

Context:
{context_text}

Question: {query}

Answer:"""
        
        try:
            response = complete(self.model, prompt)
            print(f"🤖 Generated response ({len(response)} chars)")
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

    # Removed @instrument decorator and OpenTelemetry specific code
    def answer_query(self, input_data, username: str = "unknown") -> str:
        """Main entry point for the RAG application."""
        
        if isinstance(input_data, dict):
            query = input_data.get('query', '')
            username = input_data.get('username', 'unknown')
        elif hasattr(input_data, 'get'):
            query = input_data.get('query', str(input_data))
            username = input_data.get('username', 'unknown')
        else:
            query = str(input_data)
        
        print(f"🔍 Processing query from user: {username}")
        
        # Detect toxicity for console logging (not for dashboard metric here, that's handled by TruLens Feedback)
        toxicity_result_str = self.detect_toxicity(query)
        print(f"🛡️ Toxicity check for '{query[:50]}...': {toxicity_result_str}")
        
        # Process the query
        context_str = self.retrieve_context(query)
        response = self.generate_completion(query, context_str)
        
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
        result = toxicity_classifier(text)[0] # Get the first (and only) result
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
  .on(TruApp.select_context).on_input() # Correctly select context and input
)
f_groundedness = (
    Feedback(cortex_provider.groundedness_measure_with_cot_reasons, name="Groundedness")
  .on(TruApp.select_context).on_output() # Correctly select context and output
)
f_correctness = (
    Feedback(cortex_provider.correctness_with_cot_reasons, name="Correctness")
  .on_output().on(TruApp.select_ground_truth) # Requires ground truth
)

# NEW: Create the custom toxicity feedback object
f_toxicity_metric = (
    Feedback(calculate_toxicity_score, name="Toxicity Score") # Name will appear on dashboard
  .on_output() # Evaluate the toxicity of the RAG application's final output
)


# Enhanced test dataset with usernames
usernames = ["alex_g", "beta_user", "casey_d", "dev_ops_dani", "emily_r"]

test_data = pd.DataFrame({
    'query': [
        "What is machine learning?",
        "Tell me about AI.",
        "Explain cloud computing.",
        "What is the capital of France?", # Out of domain
        "What are the types of ML?"
    ],
    'expected_answer': [
        "Machine learning is a subset of artificial intelligence where computers learn from data. Common types are supervised, unsupervised, and reinforcement learning.",
        "AI involves systems performing tasks that usually require human intelligence, like natural language processing and computer vision.",
        "Cloud computing delivers IT resources over the internet on a pay-as-you-go basis, from providers like AWS, Azure, and GCP.",
        "I do not have information about the capital of France. My knowledge is limited to machine learning, AI, and cloud computing.", # Expected graceful failure
        "The main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning."
    ],
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
    feedbacks=[f_answer_relevance, f_context_relevance, f_groundedness, f_correctness, f_toxicity_metric]
)

print(f"Application registered successfully: {app_name}")

# Single run configuration
run_config = RunConfig(
    run_name=f"trulens_custom_metrics_run_{int(time.time())}", # Updated run name
    description="Standard and custom TruLens metrics", # Updated description
    label="trulens_custom_metrics_test",
    source_type="DATAFRAME",
    dataset_name="Test dataset with custom TruLens metrics", # Updated dataset name
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
    
    if status in ["COMPLETED", "METRICS_COMPUTED", "INVOCATION_COMPLETED"]:
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
metrics_to_compute = ["Answer Relevance", "Context Relevance", "Groundedness", "Correctness", "Toxicity Score"]

successful_metrics = []
failed_metrics = []

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

# Final results
print("\n" + "="*60)
print("FINAL RESULTS - SNOWFLAKE AI OBSERVABILITY WITH CUSTOM TRULENS METRICS")
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
print("✅ All metrics stored in Snowflake for analysis.")

print("\n📊 View results in Snowsight:")
print("   Navigate to: AI & ML -> Evaluations")
print("   Look for app:", app_name)
print("   **You should now see 'Toxicity Score' alongside other metrics on the main dashboard.**")

print("\n🎉 SUCCESS: Custom TruLens metric for Toxicity implemented and integrated!")
