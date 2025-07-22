import os
import pandas as pd
import time
import warnings
import random
import re
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

# PII Detection Libraries
try:
    from transformers import pipeline
    PII_AVAILABLE = True
except ImportError:
    PII_AVAILABLE = False
    print("⚠️ Transformers pipeline not available, PII detection disabled")

# Enable TruLens OpenTelemetry tracing
os.environ["TRULENS_OTEL_TRACING"] = "1"

# Initialize detection models globally
toxicity_tokenizer = None
toxicity_model = None
hallucination_detector = None
pii_pipeline = None

# Dynamic username list
USERNAME_LIST = [
    "data_scientist_alice",
    "ml_engineer_bob", 
    "analyst_carol",
    "researcher_david",
    "ai_specialist_emma",
    "data_engineer_frank",
    "scientist_grace",
    "developer_henry"
]

def initialize_models():
    global toxicity_tokenizer, toxicity_model, hallucination_detector, pii_analyzer, pii_anonymizer
    
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
    
    # Initialize PII detection
# Initialize PII detection with transformers pipeline
    # Initialize PII detection with transformers pipeline
if PII_AVAILABLE:
    try:
        print("Loading PII detection pipeline...")
        # Use a pre-trained NER model for PII detection
        # You can replace this with your local model path: "./model"
        pii_pipeline = pipeline(
            "token-classification", 
            model="dbmdz/bert-large-cased-finetuned-conll03-english",  # ← CHANGE THIS LINE
            aggregation_strategy="first",
            device=-1  # Use CPU
        )
        
def detect_toxicity(text):
    """Detect toxicity in text - ONLY for output responses"""
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

def mask_entities(text):
    """Mask PII entities using transformers token classification pipeline"""
    global pii_pipeline
    
    if not PII_AVAILABLE or pii_pipeline is None:
        return text, False, text
    
    try:
        # Get entities from the pipeline
        entities = pii_pipeline(text, aggregation_strategy="first")
        
        if not entities:
            return text, False, text
        
        # Sort entities by start position in reverse order for proper masking
        entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        # Create masked version
        masked_text = text
        has_pii = False
        
        for entity in entities:
            # Only mask certain entity types that are considered PII
            if entity['entity_group'] in ['PERSON', 'ORG', 'LOC', 'MISC']:
                masked_text = (
                    masked_text[:entity['start']] +
                    f"[{entity['entity_group']}]" +
                    masked_text[entity['end']:]
                )
                has_pii = True
        
        return text, has_pii, masked_text
    
    except Exception as e:
        print(f"PII detection error: {e}")
        return text, False, text

def detect_and_mask_pii(text):
    """Detect PII in input query and return masked version if found"""
    return mask_entities(text)

def get_dynamic_username():
    """Get a random username from the predefined list"""
    return random.choice(USERNAME_LIST)

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
        
        # Enhanced knowledge base with diverse topics
        self.knowledge_base = {
            "machine learning": [
                "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
                "ML algorithms can identify patterns in large datasets and make predictions or decisions based on those patterns.",
                "Common types include supervised learning, unsupervised learning, and reinforcement learning.",
                "Popular ML frameworks include TensorFlow, PyTorch, and scikit-learn."
            ],
            "cloud computing": [
                "Cloud computing delivers computing services over the internet including storage, processing power, and applications.",
                "Major benefits include scalability, cost-effectiveness, and accessibility from anywhere with internet connection.",
                "Popular cloud providers include AWS, Microsoft Azure, and Google Cloud Platform.",
                "Cloud services are typically categorized as IaaS, PaaS, and SaaS."
            ],
            "artificial intelligence": [
                "AI refers to computer systems that can perform tasks typically requiring human intelligence.",
                "AI encompasses machine learning, natural language processing, computer vision, and robotics.",
                "Applications include chatbots, recommendation systems, autonomous vehicles, and medical diagnosis.",
                "AI can be classified as narrow AI, general AI, and superintelligence."
            ],
            "data science": [
                "Data science combines statistics, programming, and domain expertise to extract insights from data.",
                "The data science process includes data collection, cleaning, analysis, and visualization.",
                "Common tools include Python, R, SQL, and various visualization libraries.",
                "Data scientists help organizations make data-driven decisions."
            ],
            "cybersecurity": [
                "Cybersecurity protects digital systems, networks, and data from cyber threats.",
                "Common threats include malware, phishing, ransomware, and data breaches.",
                "Security measures include firewalls, encryption, access controls, and monitoring.",
                "Cybersecurity is critical for protecting personal and organizational information."
            ],
            "blockchain": [
                "Blockchain is a distributed ledger technology that maintains a secure record of transactions.",
                "Key features include decentralization, immutability, and transparency.",
                "Popular applications include cryptocurrencies, smart contracts, and supply chain tracking.",
                "Blockchain networks can be public, private, or consortium-based."
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
        
        # Try to find relevant topics
        for topic, contexts in self.knowledge_base.items():
            if topic in query_lower or any(keyword in query_lower for keyword in topic.split()):
                retrieved_contexts.extend(contexts)
                break
        
        # If no specific match, search for related keywords
        if not retrieved_contexts:
            keywords = ["ai", "ml", "tech", "computer", "software", "algorithm", "data"]
            if any(keyword in query_lower for keyword in keywords):
                retrieved_contexts = self.knowledge_base.get("artificial intelligence", [])
            else:
                # Default fallback
                retrieved_contexts = self.knowledge_base.get("artificial intelligence", [])
        
        return retrieved_contexts

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate_completion(self, query: str, context_str: list) -> str:
        """Generate answer from context by calling an LLM."""
        context_text = "\n".join([f"- {ctx}" for ctx in context_str])
        
        prompt = f"""Based on the following context, answer the question clearly and concisely:

Context:
{context_text}

Question: {query}

Answer:"""
        
        try:
            response = complete(self.model, prompt)
            return response
        except Exception as e:
            return f"I apologize, but I encountered an error while generating a response. Please try rephrasing your question."

    @instrument(span_type=SpanAttributes.SpanType.RECORD_ROOT)
    def answer_query(self, query: str) -> str:
        """Main entry point for the RAG application with enhanced observability."""
        # Get current span for custom attributes
        current_span = trace.get_current_span()
        
        # Get dynamic username for this query
        username = get_dynamic_username()
        current_span.set_attribute("custom.username", username)
        
        # PII Detection on input query
        original_query, has_pii, masked_query = detect_and_mask_pii(query)
        current_span.set_attribute("custom.query_has_pii", has_pii)
        current_span.set_attribute("custom.original_query", original_query)
        
        if has_pii:
            current_span.set_attribute("custom.masked_query", masked_query)
            print(f"⚠️ PII detected in query from {username}")
            print(f"Original: {original_query}")
            print(f"Masked: {masked_query}")
            # Use masked query for processing
            processing_query = masked_query
        else:
            current_span.set_attribute("custom.masked_query", "No PII detected")
            processing_query = original_query
        
        # Original RAG logic with processing query
        context_str = self.retrieve_context(processing_query)
        response = self.generate_completion(processing_query, context_str)
        
        # Toxicity detection ONLY on output response (not input query)
        response_toxicity_score, response_is_toxic = detect_toxicity(response)
        current_span.set_attribute("custom.response_toxicity_score", response_toxicity_score)
        current_span.set_attribute("custom.response_is_toxic", response_is_toxic)
        
        # Hallucination detection on response
        hallucination_score, has_hallucination = detect_hallucination(context_str, processing_query, response)
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
app_name = "enhanced_metric_rag"
tru_app = TruApp(
    test_app,
    app_name=app_name, 
    app_version="v2.0",
    connector=connector,
    main_method=test_app.answer_query
)

print(f"Application registered successfully: {app_name}")

# Enhanced test dataset with diverse question types
test_data = pd.DataFrame({
    'query': [
        "What is machine learning?",
        "How does cloud computing benefit businesses?", 
        "What are the main applications of artificial intelligence?",
        "Explain data science and its importance",
        "What is cybersecurity and why is it important?",
        "How does blockchain technology work?",
        "What are the different types of machine learning?",
        "What is the future of AI technology?",
        "How can businesses implement cloud solutions?",
        "What skills are needed for data science?"
    ],
    'expected_answer': [
        "Machine learning enables computers to learn from data without explicit programming.",
        "Cloud computing provides scalability, cost-effectiveness, and accessibility for businesses.",
        "AI applications include chatbots, recommendation systems, autonomous vehicles, and medical diagnosis.",
        "Data science combines statistics, programming, and domain expertise to extract insights from data.",
        "Cybersecurity protects digital systems, networks, and data from cyber threats and is critical for information protection.",
        "Blockchain is a distributed ledger technology that maintains secure, decentralized records of transactions.",
        "Common types include supervised learning, unsupervised learning, and reinforcement learning.",
        "AI technology continues to advance with developments in machine learning, natural language processing, and automation.",
        "Businesses can implement cloud solutions through IaaS, PaaS, and SaaS models for improved efficiency.",
        "Data science requires skills in statistics, programming (Python/R), SQL, and data visualization."
    ]
})

print(f"Created enhanced dataset with {len(test_data)} diverse test queries")

# SINGLE run configuration as per documentation
run_config = RunConfig(
    run_name="enhanced_test_run_v30",
    description="Enhanced metrics with PII detection and toxicity monitoring",
    label="enhanced_observability_test",
    source_type="DATAFRAME",
    dataset_name="Enhanced observability test dataset",
    dataset_spec={
        "RETRIEVAL.QUERY_TEXT": "query",
        "RECORD_ROOT.INPUT": "query",
        "RECORD_ROOT.GROUND_TRUTH_OUTPUT": "expected_answer",
    },
    llm_judge_name="mistral-large2"
)

print(f"Enhanced run configuration created: {run_config.run_name}")

# Add SINGLE run to TruApp
run = tru_app.add_run(run_config=run_config)
print("Enhanced run added successfully")

# Start the run and wait for completion FIRST
print("Starting enhanced run execution...")
run.start(input_df=test_data)
print("Enhanced run execution completed")

print(f"\n🔍 ENHANCED SQL QUERIES TO GET RUN DETAILS:")
print(f"============================================")
print(f"""
-- Query to get all observability data for your enhanced run
SELECT 
    timestamp,
    record_type,
    trace,
    resource_attributes,
    record_attributes,
    record
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE resource_attributes:external_agent_name::string = 'enhanced_metric_rag'
  AND resource_attributes:run_name::string = 'enhanced_test_run_v30'
ORDER BY timestamp DESC;

-- Query to get enhanced custom attributes including PII detection
SELECT 
    timestamp,
    trace:trace_id::string as trace_id,
    trace:span_id::string as span_id,
    record_attributes:"custom.username"::string as username,
    record_attributes:"custom.query_has_pii"::boolean as query_has_pii,
    record_attributes:"custom.original_query"::string as original_query,
    record_attributes:"custom.masked_query"::string as masked_query,
    record_attributes:"custom.response_toxicity_score"::float as response_toxicity_score,
    record_attributes:"custom.response_is_toxic"::boolean as response_is_toxic,
    record_attributes:"custom.hallucination_score"::float as hallucination_score,
    record_attributes:"custom.has_hallucination"::boolean as has_hallucination,
    record_attributes
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE resource_attributes:external_agent_name::string = 'enhanced_metric_rag'
  AND resource_attributes:run_name::string = 'enhanced_test_run_v30'
  AND record_type = 'SPAN'
  AND record_attributes:"custom.username" IS NOT NULL
ORDER BY timestamp DESC;

-- Query to get PII detection statistics
SELECT 
    COUNT(*) as total_queries,
    SUM(CASE WHEN record_attributes:"custom.query_has_pii"::boolean = true THEN 1 ELSE 0 END) as queries_with_pii,
    AVG(record_attributes:"custom.response_toxicity_score"::float) as avg_toxicity_score,
    SUM(CASE WHEN record_attributes:"custom.response_is_toxic"::boolean = true THEN 1 ELSE 0 END) as toxic_responses,
    SUM(CASE WHEN record_attributes:"custom.has_hallucination"::boolean = true THEN 1 ELSE 0 END) as hallucinated_responses
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE resource_attributes:external_agent_name::string = 'enhanced_metric_rag'
  AND resource_attributes:run_name::string = 'enhanced_test_run_v30'
  AND record_type = 'SPAN'
  AND record_attributes:"custom.username" IS NOT NULL;

-- Query to get evaluation metrics for the enhanced run
SELECT 
    timestamp,
    record_attributes:metric_name::string as metric_name,
    record_attributes:score::float as metric_score,
    record_attributes:explanation::string as metric_explanation
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE resource_attributes:external_agent_name::string = 'enhanced_metric_rag'
  AND resource_attributes:run_name::string = 'enhanced_test_run_v30'
  AND record_type = 'SPAN'
  AND record_attributes:metric_name IS NOT NULL
ORDER BY timestamp DESC;
""")
print(f"============================================")

# CRITICAL: Wait for invocation to complete before ANY metrics computation
print("\n" + "="*60)
print("WAITING FOR INVOCATION TO COMPLETE")
print("="*60)

max_attempts = 60
attempt = 0

while attempt < max_attempts:
    status = run.get_status()
    print(f"Attempt {attempt + 1}: Status = {status}")
    
    if status in ["INVOCATION_COMPLETED", "INVOCATION_PARTIALLY_COMPLETED"]:
        print("✅ INVOCATION COMPLETED - Ready to compute enhanced metrics!")
        break
    elif status == "INVOCATION_FAILED":
        print("❌ Invocation failed!")
        exit(1)
    else:
        time.sleep(15)
        attempt += 1

if attempt >= max_attempts:
    print("⚠️ Timeout waiting for completion, but trying metrics anyway...")

# NOW compute metrics one by one on the SAME run
print("\n" + "="*60)
print("COMPUTING ENHANCED METRICS ON SAME RUN")
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
    print(f"\n--- Computing {metric.upper()} on enhanced run ---")
    
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

# Final results
print("\n" + "="*60)
print("ENHANCED OBSERVABILITY RESULTS")
print("="*60)
print(f"✅ Successful metrics: {successful_metrics}")
print(f"❌ Failed metrics: {failed_metrics}")
print(f"Success rate: {len(successful_metrics)}/{len(metrics_to_compute)}")

final_status = run.get_status()
print(f"\nFinal enhanced run status: {final_status}")

print("\nEnhanced approach implemented:")
print("✅ Toxicity detection ONLY on output responses")
print("✅ PII detection and masking on input queries")
print("✅ Hallucination score and Boolean logging")
print("✅ Dynamic username assignment from predefined list")
print("✅ Enhanced dataset with diverse question types")
print("✅ Comprehensive custom attribute logging")

print("\n📊 Check Snowsight AI & ML -> Evaluations")
print("🔍 Enhanced run with PII detection and toxicity monitoring")

print(f"\n🎯 ENHANCED LOGGED ATTRIBUTES:")
print("✅ custom.username: [dynamic from predefined list]")
print("✅ custom.query_has_pii: [true/false]")
print("✅ custom.original_query: [original user query]")
print("✅ custom.masked_query: [PII-masked version or 'No PII detected']")
print("✅ custom.response_toxicity_score: [0.0-1.0] (OUTPUT ONLY)")
print("✅ custom.response_is_toxic: [true/false] (OUTPUT ONLY)")
print("✅ custom.hallucination_score: [0.0-1.0]")
print("✅ custom.has_hallucination: [true/false]")

print(f"\n📊 Enhanced App Name: enhanced_metric_rag")
print(f"🏃 Enhanced Run Name: enhanced_test_run_v30")
print(f"📍 Table Location: SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS")

print(f"\n🔍 Use the enhanced SQL queries above to analyze your PII-aware observability data!")
print(f"   - First query: Complete enhanced observability data")
print(f"   - Second query: Enhanced custom attributes with PII detection")  
print(f"   - Third query: PII and toxicity statistics")
print(f"   - Fourth query: Evaluation metrics results")
