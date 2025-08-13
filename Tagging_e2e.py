#!/usr/bin/env python3

import json
import time
import logging
from typing import List, Dict, Any

# For thread-based parallelization
from concurrent.futures import ThreadPoolExecutor

# LangChain components for Stage 1
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# Visualization library for Stage 4
from pyvis.network import Network

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------------------------------------------------------
# MOCK LLM FUNCTION (Synchronous Version)
# In a real scenario, this would be a blocking API call to an LLM service.
# ----------------------------------------------------------------------------
def abc_response(prompt: str) -> str:
    """
    Mocks a blocking LLM API call. It returns a pre-defined JSON string 
    based on keywords in the prompt.
    """
    logging.info("Simulating blocking LLM API call...")
    time.sleep(0.5)  # Simulate network latency

    # Mock response for Stage 2: Tag Generation
    if "Artificial intelligence is rapidly" in prompt:
        return json.dumps({
            "summary": "AI, including machine learning and NLP, is rapidly transforming industries.",
            "tags": ["AI Transformation", "Machine Learning", "NLP"],
            "entities": ["Transformer Models"],
            "intent": "Describing technological advancement"
        })
    if "ethical considerations around AI" in prompt:
        return json.dumps({
            "summary": "The deployment of AI raises significant ethical concerns like bias and data privacy.",
            "tags": ["Ethical AI", "Algorithmic Bias", "Data Privacy"],
            "entities": ["Regulatory Frameworks"],
            "intent": "Highlighting challenges and risks"
        })
    if "Climate change represents one of the most" in prompt:
        return json.dumps({
            "summary": "Climate change is a major global challenge causing severe environmental effects.",
            "tags": ["Climate Change", "Global Crisis", "Extreme Weather"],
            "entities": ["Ice Caps", "Sea Levels"],
            "intent": "Defining a global problem"
        })
    if "Renewable energy technologies are emerging" in prompt:
        return json.dumps({
            "summary": "Renewable energy technologies are key solutions to the climate crisis.",
            "tags": ["Renewable Energy", "Solar & Wind", "Energy Storage"],
            "entities": ["Solar Panels", "Wind Turbines", "Smart Grid"],
            "intent": "Proposing a solution"
        })
        
    # Mock response for Stage 3: Graph Generation
    if "You are a systems architect" in prompt:
        return json.dumps({
            "nodes": [
                {"id": "AI Transformation", "label": "AI Transformation"},
                {"id": "Ethical AI", "label": "Ethical AI"},
                {"id": "Climate Change", "label": "Climate Change"},
                {"id": "Renewable Energy", "label": "Renewable Energy"}
            ],
            "edges": [
                {"source": "AI Transformation", "target": "Ethical AI", "label": "raises"},
                {"source": "Climate Change", "target": "Renewable Energy", "label": "addressed by"}
            ]
        })
    
    return "{}"


# ----------------------------------------------------------------------------
# STAGE 1: SEMANTIC CHUNKING
# ----------------------------------------------------------------------------
def get_semantic_chunks(text: str, model_path="all-mpnet-base-v2", threshold=80) -> List[str]:
    """Chunks text semantically using LangChain."""
    logging.info(f"🤖 Loading embedding model: {model_path}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=threshold
    )
    
    logging.info(f"🔄 Chunking text with {threshold}th percentile threshold...")
    chunks = text_splitter.split_text(text)
    logging.info(f"✅ Created {len(chunks)} chunks.")
    return chunks

# ----------------------------------------------------------------------------
# STAGE 2: AUTOMATED TAG GENERATION
# ----------------------------------------------------------------------------
def generate_tags_for_chunk(chunk: str) -> Dict[str, Any]:
    """Generates tags for a single chunk by calling the LLM."""
    prompt = f"""
    You are an expert data analyst. Analyze the following text chunk and provide a JSON object that strictly follows the format below.

    # The format to follow:
    ```json
    {{
      "summary": "A one-sentence summary of the chunk's main point.",
      "tags": ["A list of 3-5 short, descriptive keywords or phrases."],
      "entities": ["A list of key named entities like technologies or organizations."],
      "intent": "A short phrase describing the primary purpose of this chunk (e.g., 'Defining a problem', 'Proposing a solution')."
    }}
    ```

    # Text Chunk to Analyze:
    {chunk}

    # Your JSON Output:
    """
    response_str = abc_response(prompt)
    try:
        data = json.loads(response_str)
        data['original_chunk'] = chunk
        return data
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from response for chunk: {chunk[:50]}...")
        return {}

# ----------------------------------------------------------------------------
# STAGE 3: TAG INTERCONNECTION & GRAPH GENERATION
# ----------------------------------------------------------------------------
def generate_system_graph(tagged_data: List[Dict]) -> Dict[str, List]:
    """Analyzes all tags and summaries to create a system-level graph."""
    context = ""
    for i, item in enumerate(tagged_data):
        tags = ", ".join(item.get('tags', []))
        context += f"Item {i+1} (Intent: {item.get('intent', 'N/A')}): Tags are [{tags}]. Summary: {item.get('summary', 'N/A')}\n"

    prompt = f"""
    You are a systems architect. Your task is to identify the relationships between the concepts described below and create a process flow graph.

    Generate a JSON object that strictly follows the format below, representing a directed graph with nodes and edges.

    # The format to follow:
    ```json
    {{
      "nodes": [
        {{"id": "unique_tag_name_1", "label": "Human-Readable Label 1"}},
        {{"id": "unique_tag_name_2", "label": "Human-Readable Label 2"}}
      ],
      "edges": [
        {{"source": "unique_tag_name_1", "target": "unique_tag_name_2", "label": "describes the relationship"}}
      ]
    }}
    ```

    # Context to Analyze:
    {context}

    # Your JSON Graph Output:
    """
    logging.info("🧠 Synthesizing system-level graph from all tags...")
    response_str = abc_response(prompt)
    try:
        return json.loads(response_str)
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON for the system graph.")
        return {"nodes": [], "edges": []}

# ----------------------------------------------------------------------------
# STAGE 4: VISUALIZATION
# ----------------------------------------------------------------------------
def create_flow_diagram(graph_data: Dict, filename: str = "flow_diagram.html"):
    """Generates an interactive HTML graph from nodes and edges data."""
    if not graph_data.get("nodes"):
        logging.warning("No nodes found in graph data. Skipping visualization.")
        return

    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", directed=True, notebook=False)
    
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    
    for node in nodes:
        net.add_node(node['id'], label=node['label'], title=node['label'])
        
    for edge in edges:
        net.add_edge(edge['source'], edge['target'], label=edge['label'], title=edge['label'])
        
    net.show_buttons(filter_=['physics'])
    net.save_graph(filename)
    logging.info(f"📈 Flow diagram saved as '{filename}'. Open this file in your browser.")

# ----------------------------------------------------------------------------
# MAIN EXECUTION PIPELINE
# ----------------------------------------------------------------------------
def main_pipeline(text: str):
    """Runs the full end-to-end pipeline using thread-based parallelization."""
    # Stage 1: Chunking (runs sequentially)
    chunks = get_semantic_chunks(text, threshold=80)
    print("\n" + "="*60)
    print("📦 SEMANTIC CHUNKS GENERATED:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk[:100]}...")
    print("="*60 + "\n")

    # Stage 2: Tag Generation (runs in parallel using threads)
    logging.info("⚙️ Starting Stage 2: Generating tags for all chunks using a thread pool...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(generate_tags_for_chunk, chunks)
        tagged_data = list(results)

    print("\n" + "="*60)
    print("🏷️ TAGS and METADATA GENERATED:")
    for i, data in enumerate(tagged_data):
        print(f"  Chunk {i+1} Tags: {data.get('tags')}")
    print("="*60 + "\n")

    # Stage 3: Graph Generation (runs sequentially)
    logging.info("⚙️ Starting Stage 3: Generating system graph...")
    graph_data = generate_system_graph(tagged_data)
    
    print("\n" + "="*60)
    print("📊 GRAPH DATA (NODES AND EDGES):")
    print(json.dumps(graph_data, indent=2))
    print("="*60 + "\n")

    # Stage 4: Visualization (runs sequentially)
    logging.info("⚙️ Starting Stage 4: Creating visualization...")
    create_flow_diagram(graph_data)

if __name__ == "__main__":
    input_paragraph = """
    Artificial intelligence is rapidly transforming industries worldwide. Machine learning algorithms are becoming increasingly sophisticated and capable of handling complex tasks. Natural language processing has seen significant breakthroughs with the advent of transformer models.

    However, ethical considerations around AI deployment are becoming increasingly important. Bias in algorithms can lead to unfair outcomes and perpetuate existing inequalities. Data privacy concerns are growing as AI systems require vast amounts of personal information.

    Climate change represents one of the most pressing challenges of our time. Rising global temperatures are causing ice caps to melt and sea levels to rise. Extreme weather events are becoming more frequent and severe.

    Renewable energy technologies are emerging as crucial solutions. Solar panels and wind turbines are becoming more efficient and cost-effective. Energy storage technologies are solving intermittency challenges.
    """
    
    main_pipeline(input_paragraph)
