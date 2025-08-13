#!/usr/bin/env python3

import re
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document

def split_sentences_manual(text):
    """Split text into sentences using regex - no NLTK needed."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def test_llamaindex_semantic_chunker(text, model_path="all-mpnet-base-v2", threshold=95):
    print(f"🤖 Loading model: {model_path}")
    
    embed_model = HuggingFaceEmbedding(
        model_name=model_path,
        device="cpu",
        embed_batch_size=16
    )
    
    # Split sentences manually first
    print("📝 Splitting sentences manually...")
    sentences = split_sentences_manual(text)
    print(f"   Found {len(sentences)} sentences")
    
    # Create documents from individual sentences
    documents = [Document(text=sentence) for sentence in sentences]
    
    # Create semantic splitter but skip internal sentence splitting
    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=1,
        breakpoint_percentile_threshold=threshold
    )
    
    print("🔄 Chunking text...")
    
    # Process pre-split sentences
    nodes = splitter.get_nodes_from_documents(documents)
    
    chunks = [node.text for node in nodes]
    
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

if __name__ == "__main__":
    paragraph = """
    Artificial intelligence is rapidly transforming industries worldwide. Machine learning algorithms are becoming increasingly sophisticated and capable of handling complex tasks. Natural language processing has seen significant breakthroughs with the advent of transformer models. Computer vision applications are now widely deployed across various sectors.
    
    However, ethical considerations around AI deployment are becoming increasingly important. Bias in algorithms can lead to unfair outcomes and perpetuate existing inequalities. Data privacy concerns are growing as AI systems require vast amounts of personal information. Regulatory frameworks are being developed to govern AI use responsibly.
    
    Climate change represents one of the most pressing challenges of our time. Rising global temperatures are causing ice caps to melt and sea levels to rise. Extreme weather events are becoming more frequent and severe. International cooperation is essential to address this global crisis effectively.
    
    Renewable energy technologies are emerging as crucial solutions. Solar panels and wind turbines are becoming more efficient and cost-effective. Energy storage technologies are solving intermittency challenges. Smart grid systems are enabling better integration of renewable sources.
    """
    
    print("📄 Input text:")
    print(paragraph.strip())
    print("\n" + "="*60)
    
    thresholds = [95, 80, 65]
    
    for threshold in thresholds:
        print(f"\n🎯 Testing with {threshold}th percentile threshold:")
        print("-" * 50)
        
        chunks = test_llamaindex_semantic_chunker(
            text=paragraph,
            model_path="all-mpnet-base-v2",
            threshold=threshold
        )
        
        print(f"\n📦 CHUNKS ({len(chunks)} total):")
        for i, chunk in enumerate(chunks, 1):
            preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
            print(f"\nChunk {i} ({len(chunk)} chars):")
            print(f"  {preview}")
        
        print(f"\n📊 Summary: {len(chunks)} chunks, avg length: {sum(len(c) for c in chunks) // len(chunks)} chars")








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
