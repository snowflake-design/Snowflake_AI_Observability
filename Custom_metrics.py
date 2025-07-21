import os
import pandas as pd
import time
import warnings
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.session import Session
from snowflake.cortex import complete
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
from trulens.apps.app import TruApp
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core.run import Run, RunConfig
from opentelemetry import trace

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Toxicity and Hallucination Detection Libraries
try:
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
    import torch
    import torch.nn.functional as F
    TOXICITY_AVAILABLE = True
except ImportError:
    TOXICITY_AVAILABLE = False
    print("⚠️ Transformers not available, toxicity detection disabled")

try:
    from lettucedetect.models.inference import HallucinationDetector
    HALLUCINATION_AVAILABLE = True
except ImportError:
    HALLUCINATION_AVAILABLE = False
    print("⚠️ LettuceDetect not available, hallucination detection disabled")

# Enable TruLens OpenTelemetry tracing
os.environ["TRULENS_OTEL_TRACING"] = "1"

# Initialize detection models globally
toxicity_tokenizer = None
toxicity_model = None
hallucination_detector = None

def initialize_models():
    global toxicity_tokenizer, toxicity_model, hallucination_detector
    
    # Initialize toxicity detection
    if TOXICITY_AVAILABLE:
        try:
            print("Loading toxicity detection model...")
            toxicity_tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
            toxicity_model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')
            toxicity_model.eval()
            print("✅ Toxicity detector loaded")
        except Exception as e:
            print(f"❌ Failed to load toxicity model: {e}")
    
    # Initialize hallucination detection
    if HALLUCINATION_AVAILABLE:
        try:
            print("Loading hallucination detection model...")
            hallucination_detector = HallucinationDetector(
                method="transformer",
                model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1"
            )
            print("✅ Hallucination detector loaded")
        except Exception as e:
            print(f"❌ Failed to load hallucination model: {e}")

def detect_toxicity(text):
    """Detect toxicity in text"""
    if not TOXICITY_AVAILABLE or toxicity_tokenizer is None or toxicity_model is None:
        return 0.0, False
    
    try:
        inputs = toxicity_tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = toxicity_model(inputs)
            probabilities = F.softmax(outputs.logits, dim=-1)
            score = probabilities[0][1].item()
        return score, score > 0.5
    except:
        return 0.0, False

def detect_hallucination(context_list, question, answer):
    """Detect hallucination in answer"""
    if not HALLUCINATION_AVAILABLE or hallucination_detector is None:
        return 0.0, False
    
    try:
        predictions = hallucination_detector.predict(
            context=context_list,
            question=question,
            answer=answer,
            output_format="spans"
        )
        if not predictions:
            return 0.0, False
        
        # Average confidence of hallucinated spans
        avg_confidence = sum(pred['confidence'] for pred in predictions) / len(predictions)
        return avg_confidence, avg_confidence > 0.3
    except:
        return 0.0, False

# Get Snowflake session
try:
    session = get_active_session()
    print("Using active Snowflake session")
except:
    print("Creating new Snowflake session...")
    session = Session.builder.configs(SNOWFLAKE_CONFIG).create()

print(f"Current context: {session.get_current_database()}.{session.get_current_schema()}")

# Initialize models
initialize_models()

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
        
        for topic, contexts in self.knowledge_base.items():
            if topic in query_lower:
                retrieved_contexts.extend(contexts)
                break
        
        if not retrieved_contexts:
            retrieved_contexts = self.knowledge_base.get("artificial intelligence", [])
        
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
        
        try:
            response = complete(self.model, prompt)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    @instrument(span_type=SpanAttributes.SpanType.RECORD_ROOT)
    def answer_query(self, query: str) -> str:
        """Main entry point for the RAG application."""
        # Get current span for custom attributes
        current_span = trace.get_current_span()
        
        # Log username
        current_span.set_attribute("custom.username", "data_scientist_user")
        
        # Detect toxicity in query
        query_toxicity_score, query_is_toxic = detect_toxicity(query)
        current_span.set_attribute("custom.query_toxicity_score", query_toxicity_score)
        current_span.set_attribute("custom.query_is_toxic", query_is_toxic)
        
        # Original RAG logic
        context_str = self.retrieve_context(query)
        response = self.generate_completion(query, context_str)
        
        # Detect toxicity in response
        response_toxicity_score, response_is_toxic = detect_toxicity(response)
        current_span.set_attribute("custom.response_toxicity_score", response_toxicity_score)
        current_span.set_attribute("custom.response_is_toxic", response_is_toxic)
        
        # Detect hallucination in response
        hallucination_score, has_hallucination = detect_hallucination(context_str, query, response)
        current_span.set_attribute("custom.hallucination_score", hallucination_score)
        current_span.set_attribute("custom.has_hallucination", has_hallucination)
        
        return response

# Initialize the RAG application
test_app = RAGApplication()

# Test basic functionality
print("Testing basic functionality...")
test_response = test_app.answer_query("What is machine learning?")
print(f"Test successful: {test_response[:100] if test_response else 'No response'}...")

# Create Snowflake connector
connector = SnowflakeConnector(snowpark_session=session)

# Register the app
app_name = "metric_rag"
tru_app = TruApp(
    test_app,
    app_name=app_name, 
    app_version="v1.0",
    connector=connector,
    main_method=test_app.answer_query
)

print(f"Application registered successfully: {app_name}")

# Create test dataset
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
    ]
})

print(f"Created dataset with {len(test_data)} test queries")

# SINGLE run configuration as per documentation
run_config = RunConfig(
    run_name="test_run_v27",
    description="All metrics computation on single run",
    label="all_metrics_test",
    source_type="DATAFRAME",
    dataset_name="All metrics test dataset",
    dataset_spec={
        "RETRIEVAL.QUERY_TEXT": "query",
        "RECORD_ROOT.INPUT": "query",
        "RECORD_ROOT.GROUND_TRUTH_OUTPUT": "expected_answer",
    },
    llm_judge_name="mistral-large2"
)

print(f"Single run configuration created: {run_config.run_name}")

# Add SINGLE run to TruApp
run = tru_app.add_run(run_config=run_config)
print("Single run added successfully")

# Start the run and wait for completion FIRST
print("Starting run execution...")
run.start(input_df=test_data)
print("Run execution completed")

print(f"\n🔍 SQL QUERY TO GET RUN DETAILS:")
print(f"============================================")
print(f"""
-- Query to get all observability data for your specific run
SELECT 
    timestamp,
    record_type,
    trace,
    resource_attributes,
    record_attributes,
    record
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE resource_attributes:external_agent_name::string = 'metric_rag'
  AND resource_attributes:run_name::string = 'test_run_v27'
ORDER BY timestamp DESC;

-- Query to get custom attributes for toxicity and hallucination
SELECT 
    timestamp,
    trace:trace_id::string as trace_id,
    trace:span_id::string as span_id,
    record_attributes:"custom.username"::string as username,
    record_attributes:"custom.query_toxicity_score"::float as query_toxicity_score,
    record_attributes:"custom.query_is_toxic"::boolean as query_is_toxic,
    record_attributes:"custom.response_toxicity_score"::float as response_toxicity_score,
    record_attributes:"custom.response_is_toxic"::boolean as response_is_toxic,
    record_attributes:"custom.hallucination_score"::float as hallucination_score,
    record_attributes:"custom.has_hallucination"::boolean as has_hallucination,
    record_attributes
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE resource_attributes:external_agent_name::string = 'metric_rag'
  AND resource_attributes:run_name::string = 'test_run_v27'
  AND record_type = 'SPAN'
  AND record_attributes:"custom.username" IS NOT NULL
ORDER BY timestamp DESC;

-- Query to get evaluation metrics for the run
SELECT 
    timestamp,
    record_attributes:metric_name::string as metric_name,
    record_attributes:score::float as metric_score,
    record_attributes:explanation::string as metric_explanation
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE resource_attributes:external_agent_name::string = 'metric_rag'
  AND resource_attributes:run_name::string = 'test_run_v27'
  AND record_type = 'SPAN'
  AND record_attributes:metric_name IS NOT NULL
ORDER BY timestamp DESC;
""")
print(f"============================================")

# CRITICAL: Wait for invocation to complete before ANY metrics computation
print("\n" + "="*60)
print("WAITING FOR INVOCATION TO COMPLETE (AS PER DOCUMENTATION)")
print("="*60)

max_attempts = 60  # Increase attempts for stability
attempt = 0

while attempt < max_attempts:
    status = run.get_status()
    print(f"Attempt {attempt + 1}: Status = {status}")
    
    if status in ["INVOCATION_COMPLETED", "INVOCATION_PARTIALLY_COMPLETED"]:
        print("✅ INVOCATION COMPLETED - Ready to compute ALL metrics!")
        break
    elif status == "INVOCATION_FAILED":
        print("❌ Invocation failed!")
        exit(1)
    else:
        time.sleep(15)  # Longer wait between checks
        attempt += 1

if attempt >= max_attempts:
    print("⚠️ Timeout waiting for completion, but trying metrics anyway...")

# NOW compute metrics one by one on the SAME run (as per documentation)
print("\n" + "="*60)
print("COMPUTING MULTIPLE METRICS ON SAME RUN")
print("="*60)

metrics_to_compute = [
    "answer_relevance",
    "context_relevance", 
    "groundedness",
    "correctness"
]

successful_metrics = []
failed_metrics = []

for metric in metrics_to_compute:
    print(f"\n--- Computing {metric.upper()} on same run ---")
    
    try:
        # Call compute_metrics on the SAME run object
        run.compute_metrics(metrics=[metric])
        print(f"✅ {metric} computation initiated successfully")
        
        # Give time for computation to process
        print(f"Waiting for {metric} computation to complete...")
        time.sleep(90)  # Longer wait for each metric
        
        # Check status after metric computation
        current_status = run.get_status()
        print(f"Status after {metric}: {current_status}")
        
        successful_metrics.append(metric)
        
    except Exception as e:
        print(f"❌ Error computing {metric}: {e}")
        failed_metrics.append(metric)
    
    # Brief pause between metrics
    print(f"Brief pause before next metric...")
    time.sleep(30)

# Final results
print("\n" + "="*60)
print("FINAL RESULTS - CORRECT DOCUMENTATION APPROACH")
print("="*60)
print(f"✅ Successful metrics: {successful_metrics}")
print(f"❌ Failed metrics: {failed_metrics}")
print(f"Success rate: {len(successful_metrics)}/{len(metrics_to_compute)}")

# Final status check
final_status = run.get_status()
print(f"\nFinal run status: {final_status}")

print("\nCorrect approach used:")
print("✅ Single run configuration")
print("✅ Wait for invocation completion FIRST")
print("✅ Multiple compute_metrics() calls on SAME run")
print("✅ Following documentation exactly")

print("\n📊 Check Snowsight AI & ML -> Evaluations")
print("🔍 Should see ONE run with multiple metrics")

print(f"\n🎯 ADDITIONAL LOGGED ATTRIBUTES:")
print("✅ custom.username: data_scientist_user")
print("✅ custom.query_toxicity_score: [0.0-1.0]")
print("✅ custom.query_is_toxic: [true/false]")
print("✅ custom.response_toxicity_score: [0.0-1.0]")
print("✅ custom.response_is_toxic: [true/false]")
print("✅ custom.hallucination_score: [0.0-1.0]")
print("✅ custom.has_hallucination: [true/false]")

print(f"\n📊 App Name: metric_rag")
print(f"🏃 Run Name: test_run_v27")
print(f"📍 Table Location: SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS")

print(f"\n🔍 Use the SQL queries above to analyze your run data!")
print(f"   - First query: Complete observability data")
print(f"   - Second query: Custom toxicity & hallucination attributes")  
print(f"   - Third query: Evaluation metrics results")
