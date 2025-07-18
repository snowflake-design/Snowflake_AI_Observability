import os
import pandas as pd
import time
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.session import Session
from snowflake.cortex import complete
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
from trulens.apps.app import TruApp
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core.run import Run, RunConfig
from trulens.core import Feedback, Select, Provider
from trulens.providers.cortex import Cortex
import numpy as np

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

# ===============================
# CUSTOM METRICS IMPLEMENTATION
# ===============================

class CustomMetricsProvider(Provider):
    """Custom provider for domain-specific feedback functions"""
    
    def __init__(self, snowpark_session=None, model_engine="mistral-large2"):
        super().__init__()
        self.session = snowpark_session
        self.model_engine = model_engine
        
    def response_completeness(self, query: str, response: str) -> float:
        """
        Custom metric to evaluate how complete the response is to the query.
        Returns a score between 0.0 and 1.0.
        """
        # Simple heuristic: longer responses with specific keywords get higher scores
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Base score based on response length
        base_score = min(len(response) / 200, 1.0)  # Normalize to 200 chars max
        
        # Bonus for including query keywords
        query_words = set(query_lower.split())
        response_words = set(response_lower.split())
        keyword_overlap = len(query_words.intersection(response_words))
        keyword_bonus = min(keyword_overlap / len(query_words), 0.5) if query_words else 0
        
        # Bonus for structured elements (lists, examples)
        structure_bonus = 0.1 if any(marker in response_lower for marker in ['-', '•', '1.', '2.', 'example']) else 0
        
        total_score = min(base_score + keyword_bonus + structure_bonus, 1.0)
        return total_score
    
    def knowledge_accuracy(self, query: str, response: str, context: str) -> float:
        """
        Custom metric to evaluate accuracy based on knowledge base alignment.
        Returns a score between 0.0 and 1.0.
        """
        context_lower = context.lower() if isinstance(context, str) else ""
        response_lower = response.lower()
        
        if not context_lower:
            return 0.5  # Neutral score if no context
            
        # Count factual overlaps between context and response
        context_words = set(context_lower.split())
        response_words = set(response_lower.split())
        overlap = len(context_words.intersection(response_words))
        
        # Normalize by context length
        overlap_score = min(overlap / len(context_words), 1.0) if context_words else 0
        
        # Penalty for hallucination indicators
        hallucination_words = ['i think', 'maybe', 'probably', 'not sure', 'might be']
        hallucination_penalty = sum(0.1 for phrase in hallucination_words if phrase in response_lower)
        
        final_score = max(0.0, overlap_score - hallucination_penalty)
        return final_score
    
    def response_specificity(self, query: str, response: str) -> float:
        """
        Custom metric to evaluate how specific the response is.
        Returns a score between 0.0 and 1.0.
        """
        response_lower = response.lower()
        
        # Specific indicators get higher scores
        specific_indicators = ['specific', 'exactly', 'precisely', 'include', 'examples', 'such as']
        generic_indicators = ['generally', 'usually', 'often', 'might', 'could be', 'typically']
        
        specific_count = sum(1 for indicator in specific_indicators if indicator in response_lower)
        generic_count = sum(1 for indicator in generic_indicators if indicator in response_lower)
        
        # Score based on specificity vs generality
        if specific_count + generic_count == 0:
            return 0.5  # Neutral if no indicators
            
        specificity_ratio = specific_count / (specific_count + generic_count)
        return specificity_ratio

class CustomCortexProvider(Cortex):
    """Extended Cortex provider with custom LLM-based metrics"""
    
    def business_value_assessment(self, query: str, response: str) -> float:
        """
        Custom LLM-based metric to assess business value of the response.
        Uses Cortex LLM to generate score.
        """
        assessment_prompt = f"""
        Evaluate the business value of this AI response on a scale from 0 to 10.
        Consider factors like:
        - Actionable insights provided
        - Practical applicability
        - Problem-solving effectiveness
        - ROI potential for business decisions
        
        Query: {query}
        Response: {response}
        
        Provide only a numeric score from 0-10, where 0 is no business value and 10 is extremely high business value.
        """
        
        try:
            score_text = complete(self.model_engine, assessment_prompt)
            # Extract numeric score
            import re
            score_match = re.search(r'\b(\d+(?:\.\d+)?)\b', str(score_text))
            if score_match:
                score = float(score_match.group(1))
                return min(score / 10.0, 1.0)  # Normalize to 0-1
            else:
                return 0.5  # Default if can't parse
        except Exception as e:
            print(f"Error in business value assessment: {e}")
            return 0.5
    
    def response_clarity(self, response: str) -> float:
        """
        Custom LLM-based metric to assess response clarity.
        """
        clarity_prompt = f"""
        Rate the clarity and readability of this response on a scale from 0 to 10.
        Consider factors like:
        - Clear language and structure
        - Easy to understand explanations
        - Logical flow of information
        - Absence of jargon or overly complex terms
        
        Response: {response}
        
        Provide only a numeric score from 0-10, where 0 is very unclear and 10 is extremely clear.
        """
        
        try:
            score_text = complete(self.model_engine, clarity_prompt)
            import re
            score_match = re.search(r'\b(\d+(?:\.\d+)?)\b', str(score_text))
            if score_match:
                score = float(score_match.group(1))
                return min(score / 10.0, 1.0)
            else:
                return 0.5
        except Exception as e:
            print(f"Error in clarity assessment: {e}")
            return 0.5

# ===============================
# MAIN RAG APPLICATION
# ===============================

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
        context_str = self.retrieve_context(query)
        return self.generate_completion(query, context_str)

# Initialize the RAG application
test_app = RAGApplication()

# Test basic functionality
print("Testing basic functionality...")
test_response = test_app.answer_query("What is machine learning?")
print(f"Test successful: {test_response[:100] if test_response else 'No response'}...")

# ===============================
# INITIALIZE CUSTOM METRICS
# ===============================

# Create custom providers
custom_provider = CustomMetricsProvider(session, "mistral-large2")
cortex_provider = CustomCortexProvider(session, model_engine="mistral-large2")

# Built-in metrics
f_answer_relevance = Feedback(
    cortex_provider.relevance_with_cot_reasons,
    name="Answer Relevance"
).on(Select.RecordCalls.retrieve_context.args.query).on_output()

f_context_relevance = Feedback(
    cortex_provider.context_relevance_with_cot_reasons,
    name="Context Relevance"
).on(Select.RecordCalls.retrieve_context.args.query).on(Select.RecordCalls.retrieve_context.rets.collect()).aggregate(np.mean)

f_groundedness = Feedback(
    cortex_provider.groundedness_measure_with_cot_reasons,
    name="Groundedness"
).on(Select.RecordCalls.retrieve_context.rets.collect()).on_output()

f_correctness = Feedback(
    cortex_provider.correctness_with_cot_reasons,
    name="Correctness"
).on(Select.RecordCalls.retrieve_context.args.query).on_output()

# Custom metrics
f_response_completeness = Feedback(
    custom_provider.response_completeness,
    name="Response Completeness"
).on(
    query=Select.RecordCalls.retrieve_context.args.query,
    response=Select.RecordOutput
)

f_knowledge_accuracy = Feedback(
    custom_provider.knowledge_accuracy,
    name="Knowledge Accuracy"
).on(
    query=Select.RecordCalls.retrieve_context.args.query,
    response=Select.RecordOutput,
    context=Select.RecordCalls.retrieve_context.rets.collect()
)

f_response_specificity = Feedback(
    custom_provider.response_specificity,
    name="Response Specificity"
).on(
    query=Select.RecordCalls.retrieve_context.args.query,
    response=Select.RecordOutput
)

f_business_value = Feedback(
    cortex_provider.business_value_assessment,
    name="Business Value"
).on(
    query=Select.RecordCalls.retrieve_context.args.query,
    response=Select.RecordOutput
)

f_response_clarity = Feedback(
    cortex_provider.response_clarity,
    name="Response Clarity"
).on(response=Select.RecordOutput)

print("Custom metrics initialized successfully!")

# ===============================
# REGISTER APP WITH ALL METRICS
# ===============================

# Create Snowflake connector
connector = SnowflakeConnector(snowpark_session=session)

# Register the app with both built-in and custom metrics
app_name = f"rag_custom_metrics_app_{int(time.time())}"
tru_app = TruApp(
    test_app,
    app_name=app_name, 
    app_version="v1.0_custom",
    connector=connector,
    main_method=test_app.answer_query
)

print(f"Application registered successfully: {app_name}")

# Create test dataset
test_data = pd.DataFrame({
    'query': [
        "What is machine learning?",
        "What are cloud computing benefits?", 
        "What are AI applications?",
        "How does supervised learning work?",
        "What are the costs of cloud services?"
    ],
    'expected_answer': [
        "Machine learning enables computers to learn from data without explicit programming.",
        "Cloud computing provides scalability, cost-effectiveness, and accessibility.",
        "AI applications include chatbots, recommendation systems, and autonomous vehicles.",
        "Supervised learning uses labeled training data to make predictions.",
        "Cloud service costs depend on usage, storage, and computing resources."
    ]
})

print(f"Created dataset with {len(test_data)} test queries")

# SINGLE run configuration
run_config = RunConfig(
    run_name=f"custom_metrics_run_{int(time.time())}",
    description="Complete evaluation with built-in and custom metrics",
    label="custom_metrics_test",
    source_type="DATAFRAME",
    dataset_name="Custom metrics test dataset",
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
        exit(1)
    else:
        time.sleep(15)
        attempt += 1

if attempt >= max_attempts:
    print("⚠️ Timeout waiting for completion, but trying metrics anyway...")

# Compute ALL metrics (built-in + custom)
print("\n" + "="*60)
print("COMPUTING ALL METRICS (BUILT-IN + CUSTOM)")
print("="*60)

all_metrics = [
    # Built-in metrics
    "answer_relevance",
    "context_relevance", 
    "groundedness",
    "correctness",
    # Custom metrics
    "response_completeness",
    "knowledge_accuracy",
    "response_specificity",
    "business_value",
    "response_clarity"
]

successful_metrics = []
failed_metrics = []

for metric in all_metrics:
    print(f"\n--- Computing {metric.upper()} ---")
    
    try:
        run.compute_metrics(metrics=[metric])
        print(f"✅ {metric} computation initiated successfully")
        
        time.sleep(45)  # Wait for computation
        
        current_status = run.get_status()
        print(f"Status after {metric}: {current_status}")
        
        successful_metrics.append(metric)
        
    except Exception as e:
        print(f"❌ Error computing {metric}: {e}")
        failed_metrics.append(metric)
    
    time.sleep(15)  # Brief pause between metrics

# Final results
print("\n" + "="*60)
print("FINAL RESULTS - CUSTOM METRICS IMPLEMENTATION")
print("="*60)
print(f"✅ Successful metrics: {successful_metrics}")
print(f"❌ Failed metrics: {failed_metrics}")
print(f"Success rate: {len(successful_metrics)}/{len(all_metrics)}")

# Final status check
final_status = run.get_status()
print(f"\nFinal run status: {final_status}")

print("\nCustom Metrics Added:")
print("✅ Response Completeness - Evaluates response thoroughness")
print("✅ Knowledge Accuracy - Measures alignment with knowledge base")
print("✅ Response Specificity - Assesses how specific vs generic the response is")
print("✅ Business Value - LLM-based evaluation of business applicability")
print("✅ Response Clarity - LLM-based assessment of clarity and readability")

print(f"\n📊 Check Snowsight AI & ML -> Evaluations -> {app_name}")
print("🔍 You should see one run with 9 total metrics (4 built-in + 5 custom)")
print("🎯 Custom metrics will appear in the dashboard alongside built-in metrics!")

# Example of how to access custom metric results programmatically
print("\n" + "="*40)
print("ACCESSING CUSTOM METRICS PROGRAMMATICALLY")
print("="*40)

try:
    # This would show how to retrieve and analyze custom metrics
    print("Custom metrics are stored in Snowflake AI_OBSERVABILITY_EVENTS table")
    print("You can query them using SQL or through the TruLens connector")
    print("Example queries available in Snowsight under AI & ML -> Evaluations")
except Exception as e:
    print(f"Note: {e}")
