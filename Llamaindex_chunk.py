#!/usr/bin/env python3

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document

def semantic_chunk_with_llamaindex(text, model_path="all-MiniLM-L6-v2", threshold=95):
    print(f"🤖 Loading model: {model_path}")
    
    embed_model = HuggingFaceEmbedding(
        model_name=model_path,  # Can be model name OR local path
        device="cpu",
        embed_batch_size=16
    )
    
    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=1,
        breakpoint_percentile_threshold=threshold
    )
    
    print("🔄 Chunking text...")
    
    document = Document(text=text)
    nodes = splitter.get_nodes_from_documents([document])
    
    chunks = [node.text for node in nodes]
    
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# Example usage
if __name__ == "__main__":
    paragraph = """
    Artificial intelligence is rapidly advancing across multiple domains. Machine learning algorithms are becoming more sophisticated and capable of handling complex tasks. Natural language processing has seen significant breakthroughs with transformer models. Computer vision applications are now widely deployed in various industries. However, ethical considerations around AI deployment are increasingly important. Bias in algorithms can lead to unfair outcomes. Data privacy concerns are growing as AI systems require vast amounts of personal information. Regulatory frameworks are being developed to govern AI use responsibly. The future of AI depends on addressing these challenges while continuing technological progress.
    """
    
    print("📄 Input text:")
    print(paragraph.strip())
    print("\n" + "="*60)
    
    # Option 1: Use model name
    chunks = semantic_chunk_with_llamaindex(
        text=paragraph,
        model_path="all-MiniLM-L6-v2",
        threshold=80
    )
    
    # Option 2: Use local path (uncomment to use)
    # chunks = semantic_chunk_with_llamaindex(
    #     text=paragraph,
    #     model_path="/path/to/your/local/model",  # Local model directory
    #     threshold=80
    # )
    
    print(f"\n📦 SEMANTIC CHUNKS:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  {chunk}")
    
    print(f"\n📊 Summary: {len(chunks)} chunks created")
