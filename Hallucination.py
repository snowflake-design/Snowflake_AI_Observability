"""
Simple HDM-2 Hallucination Detection Usage
==========================================
Install: pip install hdm2
"""

from hdm2 import HallucinationDetectionModel

# Load model
model = HallucinationDetectionModel()

# Example 1: Basic usage
prompt = "Tell me about penguins"
context = "Penguins are flightless birds that live in Antarctica and other cold regions."
response = "Penguins are flightless birds that can fly short distances when threatened."

results = model.apply(prompt=prompt, context=context, response=response)

print("Hallucination detected:", results['hallucination_detected'])
print("Severity score:", results['hallucination_severity'])
print("Problematic words:", [w['text'] for w in results['high_scoring_words']])

# Example 2: Medical context
prompt = "What are our hospital's Q1 2025 trial results?"
context = """
In Q1 2025, Northbridge Medical Center enrolled 220 patients in the ORION-5 oncology study.
The cardiology trials enrolled 145 patients total.
"""
response = """
In Q1 2025, Northbridge enrolled 573 patients across four trials. 
ORION-5 had the highest enrollment with 220 patients.
Northbridge has led regional enrollments since 2020 in oncology research.
"""

results = model.apply(prompt=prompt, context=context, response=response)

print("\nMedical Example:")
print("Hallucination detected:", results['hallucination_detected'])
print("Severity:", f"{results['hallucination_severity']:.3f}")
if results['candidate_sentences']:
    print("Problematic sentences:", results['candidate_sentences'])

# Example 3: Financial context
prompt = "What were our revenue figures?"
context = "Company Q4 revenue was $2.1 million with 15% growth year-over-year."
response = "Our Q4 revenue reached $3.5 million, representing 40% growth and breaking all previous records."

results = model.apply(prompt=prompt, context=context, response=response)

print("\nFinancial Example:")
print("Hallucination detected:", results['hallucination_detected'])
print("Severity:", f"{results['hallucination_severity']:.3f}")

# Example 4: Custom thresholds for sensitive domains
results = model.apply(
    prompt=prompt,
    context=context, 
    response=response,
    token_threshold=0.3,  # Lower threshold = more sensitive
    ck_threshold=0.6
)

print("\nWith custom thresholds:")
print("Hallucination detected:", results['hallucination_detected'])
