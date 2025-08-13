#!/usr/bin/env python3

import re
from semantic_chunker.core import SemanticChunker

def split_into_sentences(text):
    """Split text into sentences first."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [{"text": s.strip()} for s in sentences if s.strip()]

def test_advanced_chunker(text, model_path="all-mpnet-base-v2", threshold=0.4):
    print(f"🤖 Loading model: {model_path}")
    
    chunker = SemanticChunker(
        model_name=model_path,
        max_tokens=512,
        cluster_threshold=threshold,
        similarity_threshold=0.4
    )
    
    print("📝 Splitting text into sentences...")
    
    # IMPORTANT: Split text into sentences first!
    chunks = split_into_sentences(text)
    print(f"   Found {len(chunks)} sentences")
    
    print("🔄 Chunking text...")
    
    merged_chunks = chunker.chunk(chunks)
    
    print(f"✅ Created {len(merged_chunks)} chunks")
    return merged_chunks

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
    
    thresholds = [0.5, 0.4, 0.3]
    
    for threshold in thresholds:
        print(f"\n🎯 Testing with {threshold} cluster threshold:")
        print("-" * 50)
        
        chunks = test_advanced_chunker(
            text=paragraph,
            model_path="all-mpnet-base-v2",
            threshold=threshold
        )
        
        print(f"\n📦 CHUNKS ({len(chunks)} total):")
        for i, chunk in enumerate(chunks, 1):
            text_content = chunk["text"]
            preview = text_content[:100] + "..." if len(text_content) > 100 else text_content
            print(f"\nChunk {i} ({len(text_content)} chars):")
            print(f"  {preview}")
        
        print(f"\n📊 Summary: {len(chunks)} chunks")








#!/usr/bin/env python3

from semantic_chunker.core import SemanticChunker

def test_advanced_chunker(text, model_path="all-mpnet-base-v2", threshold=0.4):
    print(f"🤖 Loading model: {model_path}")
    
    chunker = SemanticChunker(
        model_name=model_path,
        max_tokens=512,
        cluster_threshold=threshold,
        similarity_threshold=0.4
    )
    
    print("🔄 Chunking text...")
    
    chunks = [{"text": text}]
    merged_chunks = chunker.chunk(chunks)
    
    print(f"✅ Created {len(merged_chunks)} chunks")
    return merged_chunks

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
    
    thresholds = [0.5, 0.4, 0.3]
    
    for threshold in thresholds:
        print(f"\n🎯 Testing with {threshold} cluster threshold:")
        print("-" * 50)
        
        chunks = test_advanced_chunker(
            text=paragraph,
            model_path="all-mpnet-base-v2",
            threshold=threshold
        )
        
        print(f"\n📦 CHUNKS ({len(chunks)} total):")
        for i, chunk in enumerate(chunks, 1):
            text_content = chunk["text"]
            preview = text_content[:100] + "..." if len(text_content) > 100 else text_content
            print(f"\nChunk {i} ({len(text_content)} chars):")
            print(f"  {preview}")
        
        print(f"\n📊 Summary: {len(chunks)} chunks")
