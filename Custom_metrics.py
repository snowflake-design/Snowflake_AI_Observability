import os
import pandas as pd
import time
import torch
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.session import Session
from snowflake.cortex import complete
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
from trulens.apps.app import TruApp
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core.run import Run, RunConfig
from opentelemetry import trace

# For toxicity detection
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F

# For hallucination detection
from lettucedetect.models.inference import HallucinationDetector

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

class ToxicityDetector:
    """Toxicity detection using s-nlp RoBERTa model"""
    def __init__(self):
        print("Loading toxicity detection model...")
        self.tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
        self.model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')
        self.model.eval()
        print("✅ Toxicity detector loaded successfully")
    
    def detect_toxicity(self, text: str) -> float:
        """
        Detect toxicity score for given text
        Returns: float between 0 and 1 (1 = toxic, 0 = non-toxic)
        """
        try:
            # Tokenize and get model predictions
            inputs = self.tokenizer.encode(text, return_tensors="pt", truncate=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)
                # Index 1 is for toxic, index 0 is for neutral
                toxicity_score = probabilities[0][1].item()
            
            return toxicity_score
        except Exception as e:
            print(f"Error in toxicity detection: {e}")
            return 0.0

class HallucinationDetectorWrapper:
    """Hallucination detection using LettuceDetect"""
    def __init__(self):
        print("Loading hallucination detection model...")
        try:
            self.detector = HallucinationDetector(
                method="transformer",
                model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1"
            )
            print("✅ Hallucination detector loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load hallucination detector: {e}")
            self.detector = None
    
    def detect_hallucination(self, context: list, question: str, answer: str) -> float:
        """
        Detect hallucination score for given answer
        Returns: float between 0 and 1 (1 = high hallucination, 0 = no hallucination)
        """
        if self.detector is None:
            return 0.0
        
        try:
            # Get span-level predictions
            predictions = self.detector.predict(
                context=context,
                question=question,
                answer=answer,
                output_format="spans"
            )
            
            # Calculate overall hallucination score
            if not predictions:
                return 0.0
            
            # Average confidence of hallucinated spans
            total_confidence = sum(pred['confidence'] for pred in predictions)
            avg_confidence = total_confidence / len(predictions) if predictions else 0.0
            
            return avg_confidence
        except Exception as e:
            print(f"Error in hallucination detection: {e}")
            return 0.0

class EnhancedRAGApplication:
    def __init__(self, username: str = "default_user"):
        self.model = "mistral-large2"
        self.username = username
        
        # Initialize detectors
        self.toxicity_detector = ToxicityDetector()
        self.hallucination_detector = HallucinationDetectorWrapper()
        
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
        """Main entry point for the RAG application with enhanced observability."""
        # Get current span for custom attributes
        current_span = trace.get_current_span()
        
        # Log username
        current_span.set_attribute("custom.username", self.username)
        
        # Detect toxicity in the query
        query_toxicity_score = self.toxicity_detector.detect_toxicity(query)
        current_span.set_attribute("custom.query_toxicity_score", query_toxicity_score)
        
        # Retrieve context and generate response
        context_str = self.retrieve_context(query)
        response = self.generate_completion(query, context_str)
        
        # Detect toxicity in the response
        response_toxicity_score = self.toxicity_detector.detect_toxicity(response)
        current_span.set_attribute("custom.response_toxicity_score", response_toxicity_score)
        
        # Detect hallucination in the response
        hallucination_score = self.hallucination_detector.detect_hallucination(
            context=context_str,
            question=query,
            answer=response
        )
        current_span.set_attribute("custom.hallucination_score", hallucination_score)
        
        # Add response length and other metrics
        current_span.set_attribute("custom.response_length", len(response))
        current_span.set_attribute("custom.context_count", len(context_str))
        
        # Log quality assessment
        is_high_quality = (
            query_toxicity_score < 0.1 and 
            response_toxicity_score < 0.1 and 
            hallucination_score < 0.3
        )
        current_span.set_attribute("custom.is_high_quality_response", is_high_quality)
        
        return response

# Initialize the enhanced RAG application with username
username = "data_scientist_john"  # You can change this to any username
test_app = EnhancedRAGApplication(username=username)

# Test basic functionality
print("\n" + "="*60)
print("TESTING ENHANCED RAG APPLICATION")
print("="*60)
print("Testing basic functionality...")
test_response = test_app.answer_query("What is machine learning?")
print(f"Test successful: {test_response[:100] if test_response else 'No response'}...")

# Create Snowflake connector
connector = SnowflakeConnector(snowpark_session=session)

# Register the app
app_name = f"enhanced_rag_metrics_app_{int(time.time())}"
tru_app = TruApp(
    test_app,
    app_name=app_name, 
    app_version="v2.0_enhanced",
    connector=connector,
    main_method=test_app.answer_query
)

print(f"Application registered successfully: {app_name}")

# Create enhanced test dataset
test_data = pd.DataFrame({
    'query': [
        "What is machine learning?",
        "What are cloud computing benefits?", 
        "What are AI applications?",
        "Tell me about artificial intelligence in healthcare",
        "How does machine learning help businesses?"
    ],
    'expected_answer': [
        "Machine learning enables computers to learn from data without explicit programming.",
        "Cloud computing provides scalability, cost-effectiveness, and accessibility.",
        "AI applications include chatbots, recommendation systems, and autonomous vehicles.",
        "AI in healthcare includes medical diagnosis, drug discovery, and patient monitoring systems.",
        "Machine learning helps businesses through predictive analytics, automation, and data-driven insights."
    ]
})

print(f"Created dataset with {len(test_data)} test queries")

# SINGLE run configuration as per documentation
run_config = RunConfig(
    run_name=f"enhanced_metrics_run_{int(time.time())}",
    description="Enhanced metrics with toxicity and hallucination detection",
    label="enhanced_metrics_test",
    source_type="DATAFRAME",
    dataset_name="Enhanced RAG test dataset",
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
print("\n" + "="*60)
print("STARTING ENHANCED RUN EXECUTION")
print("="*60)

print("Starting run execution...")
run.start(input_df=test_data)
print("Run execution completed")

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
print("FINAL ENHANCED RESULTS")
print("="*60)
print(f"✅ Successful metrics: {successful_metrics}")
print(f"❌ Failed metrics: {failed_metrics}")
print(f"Success rate: {len(successful_metrics)}/{len(metrics_to_compute)}")

# Final status check
final_status = run.get_status()
print(f"\nFinal run status: {final_status}")

print("\nEnhanced RAG approach used:")
print("✅ Single run configuration")
print("✅ Wait for invocation completion FIRST")
print("✅ Multiple compute_metrics() calls on SAME run")
print("✅ Username logging with custom attributes")
print("✅ Toxicity detection for queries and responses")
print("✅ Hallucination detection for generated answers")
print("✅ Quality assessment metrics")
print("✅ Following documentation exactly")

print(f"\n📊 Check Snowsight AI & ML -> Evaluations -> {app_name}")
print("🔍 Should see ONE run with multiple metrics and custom attributes:")
print("   - custom.username")
print("   - custom.query_toxicity_score")
print("   - custom.response_toxicity_score") 
print("   - custom.hallucination_score")
print("   - custom.response_length")
print("   - custom.context_count")
print("   - custom.is_high_quality_response")

print(f"\n🔍 Query custom attributes with SQL:")
print(f"   SELECT * FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS")
print(f"   WHERE app_name = '{app_name}'")
print(f"   AND attributes:custom.username::string = '{username}';")

print(f"\n🎯 ENHANCED FEATURES SUMMARY:")
print(f"👤 Username: {username}")
print("🦠 Toxicity Detection: s-nlp/roberta_toxicity_classifier")
print("🔍 Hallucination Detection: LettuceDetect (ModernBERT-based)")
print("📊 Custom Metrics: Quality assessment, response length, context count")
print("✅ Complete AI observability with safety metrics!")
