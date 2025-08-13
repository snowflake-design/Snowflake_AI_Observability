#!/usr/bin/env python3

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

def test_langchain_semantic_chunker(text, model_path="all-mpnet-base-v2", threshold=95):
    print(f"🤖 Loading model: {model_path}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=threshold,
        buffer_size=1
    )
    
    print("🔄 Chunking text...")
    
    chunks = text_splitter.split_text(text)
    
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# Example usage
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
        
        chunks = test_langchain_semantic_chunker(
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
