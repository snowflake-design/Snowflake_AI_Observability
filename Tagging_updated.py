import streamlit as st
import pickle
import hashlib
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
from pathlib import Path

# LlamaIndex imports
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# PDF processing
import PyPDF2

# Streamlit Flow
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from uuid import uuid4

# Set page config
st.set_page_config(
    page_title="🏷️ Semantic Document Tagger",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .chunk-preview {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .tag-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    
    .processing-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .success-message {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Placeholder function for LLM response (replace with your actual implementation)
def abc_response(prompt: str) -> str:
    """
    Placeholder for your LLM function
    Replace this with your actual LLM implementation
    """
    # This is a mock response - replace with your actual LLM call
    if "hierarchy" in prompt.lower():
        return json.dumps({
            "nodes": [
                {
                    "id": "root",
                    "label": "Document Overview",
                    "level": 0,
                    "parent": None,
                    "confidence": 0.95,
                    "chunks": [],
                    "node_type": "input",
                    "position": {"x": 250, "y": 50}
                }
            ],
            "edges": []
        })
    else:
        return "sample_tag, another_tag, third_tag"

class DocumentProcessor:
    def __init__(self):
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize embedding model (all-MiniLM-L6-v2 local model)
        if 'embed_model' not in st.session_state:
            with st.spinner("Loading embedding model..."):
                st.session_state.embed_model = HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    cache_folder="./models"
                )
        
        self.embed_model = st.session_state.embed_model
        
        # Initialize semantic splitter
        self.semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=self.embed_model
        )
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def get_file_hash(self, content: str) -> str:
        """Generate hash for content caching"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def load_from_cache(self, file_hash: str) -> Dict[str, Any]:
        """Load processed results from cache"""
        cache_file = self.cache_dir / f"{file_hash}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_to_cache(self, file_hash: str, data: Dict[str, Any]):
        """Save processed results to cache"""
        cache_file = self.cache_dir / f"{file_hash}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    
    def semantic_chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into semantic chunks using LlamaIndex"""
        try:
            # Create document
            document = Document(text=text)
            
            # Get semantic nodes
            nodes = self.semantic_splitter.get_nodes_from_documents([document])
            
            # Convert to our format
            chunks = []
            for i, node in enumerate(nodes):
                chunks.append({
                    'id': f'chunk_{i+1}',
                    'text': node.text,
                    'metadata': node.metadata,
                    'start_char_idx': getattr(node, 'start_char_idx', None),
                    'end_char_idx': getattr(node, 'end_char_idx', None)
                })
            
            return chunks
        except Exception as e:
            st.error(f"Error in semantic chunking: {str(e)}")
            return []
    
    def generate_tags_for_chunk(self, chunk: Dict[str, Any]) -> List[str]:
        """Generate tags for a single chunk using LLM"""
        prompt = f"""
        Analyze the following text chunk and generate comprehensive tags that capture:
        1. Major topics/themes discussed
        2. Key entities mentioned  
        3. Content types/categories
        4. Processes or actions described
        5. Emotional tone or sentiment
        6. Technical terms or concepts
        7. Geographic locations
        8. Temporal references
        9. Relationships between entities
        10. Any other relevant semantic information

        Generate as many relevant tags as needed to fully capture the chunk's content.
        
        Text chunk:
        {chunk['text'][:500]}...
        
        Return only the tags as a comma-separated list (no explanations):
        """
        
        try:
            response = abc_response(prompt)
            tags = [tag.strip() for tag in response.split(',') if tag.strip()]
            return tags  # No limit on number of tags
        except Exception as e:
            st.error(f"Error generating tags: {str(e)}")
            return []
    
    def generate_tags_parallel(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Generate tags for all chunks in parallel"""
        chunk_tags = {}
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(self.generate_tags_for_chunk, chunk): chunk['id'] 
                for chunk in chunks
            }
            
            # Collect results as they complete
            progress_bar = st.progress(0)
            completed = 0
            total = len(chunks)
            
            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    tags = future.result()
                    chunk_tags[chunk_id] = tags
                except Exception as e:
                    st.error(f"Error processing chunk {chunk_id}: {str(e)}")
                    chunk_tags[chunk_id] = []
                
                completed += 1
                progress_bar.progress(completed / total)
        
        return chunk_tags
    
    def create_tag_hierarchy(self, chunk_tags: Dict[str, List[str]], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create hierarchical tag structure using LLM"""
        # Collect all unique tags
        all_tags = set()
        for tags in chunk_tags.values():
            all_tags.update(tags)
        
        # Create tag-to-chunks mapping
        tag_chunk_mapping = {}
        for chunk_id, tags in chunk_tags.items():
            for tag in tags:
                if tag not in tag_chunk_mapping:
                    tag_chunk_mapping[tag] = []
                tag_chunk_mapping[tag].append(chunk_id)
        
        prompt = f"""
        You are helping analyze a document by organizing tags that were extracted from semantic chunks. 
        
        **CONTEXT & GOAL:**
        We semantically chunked a document and generated tags for each chunk to understand the document's structure and content. 
        Your task is to "connect the dots" by creating a meaningful hierarchy that reveals:
        - Main themes and sub-topics in the document
        - Relationships between different concepts
        - Document structure and information flow
        - Key entities and their roles
        - Processes, procedures, or workflows described
        
        **DOCUMENT ANALYSIS TASK:**
        These tags represent the semantic content of document chunks. Create a hierarchy that helps users:
        1. Navigate the document's conceptual structure
        2. Understand relationships between topics
        3. Discover the document's main themes and supporting details
        4. See how different parts of the document connect
        
        **EXTRACTED TAGS:** {', '.join(all_tags)}
        
        **CHUNK-TAG MAPPING:**
        {json.dumps({tag: chunks for tag, chunks in tag_chunk_mapping.items()}, indent=2)}
        
        Create a meaningful hierarchy that tells the story of this document. Return a JSON structure suitable for Streamlit Flow with the following format:
        {{
            "nodes": [
                {{
                    "id": "unique_id",
                    "label": "Tag Name",
                    "level": 0,
                    "parent": null,
                    "confidence": 0.95,
                    "chunks": ["chunk_1", "chunk_2"],
                    "node_type": "input|default|output",
                    "position": {{"x": 250, "y": 50}}
                }}
            ],
            "edges": [
                {{
                    "id": "parent-child",
                    "source": "parent_id",
                    "target": "child_id",
                    "relationship": "parent-child"
                }}
            ]
        }}
        
        Rules:
        - Create a clear hierarchy with 3-4 levels maximum
        - Root nodes should be broad categories (node_type: "input")
        - Leaf nodes should be specific tags (node_type: "output") 
        - Middle nodes use "default"
        - Position nodes in a tree layout (x increases by level, y varies by siblings)
        - Include the chunks associated with each tag
        
        Return only valid JSON:
        """
        
        try:
            response = abc_response(prompt)
            hierarchy = json.loads(response)
            
            # Add chunk associations to nodes
            for node in hierarchy.get('nodes', []):
                tag_name = node['label'].lower().replace(' ', '_')
                if tag_name in tag_chunk_mapping:
                    node['chunks'] = tag_chunk_mapping[tag_name]
                else:
                    # Try to find partial matches
                    matching_chunks = []
                    for tag, chunks in tag_chunk_mapping.items():
                        if tag.lower() in tag_name or tag_name in tag.lower():
                            matching_chunks.extend(chunks)
                    node['chunks'] = list(set(matching_chunks))
            
            return hierarchy
        except Exception as e:
            st.error(f"Error creating hierarchy: {str(e)}")
            return {"nodes": [], "edges": []}

def convert_to_streamlit_flow(hierarchy: Dict[str, Any], chunks: List[Dict[str, Any]]) -> tuple:
    """Convert hierarchy to Streamlit Flow format"""
    nodes = []
    edges = []
    
    # Create chunk lookup for hover information
    chunk_lookup = {chunk['id']: chunk for chunk in chunks}
    
    for node_data in hierarchy.get('nodes', []):
        # Color based on level
        level = node_data.get('level', 0)
        if level == 0:
            color = '#2E86AB'  # Blue for root
        elif level == 1:
            color = '#A23B72'  # Purple for categories
        elif level == 2:
            color = '#F18F01'  # Orange for subcategories
        else:
            color = '#C73E1D'  # Red for specific tags
        
        # Size based on number of chunks
        chunk_count = len(node_data.get('chunks', []))
        size_factor = max(60, min(120, 60 + chunk_count * 10))
        
        node = StreamlitFlowNode(
            id=f"tag-{node_data['id']}",
            pos=(node_data['position']['x'], node_data['position']['y']),
            data={
                'content': f"{node_data['label']}\n({node_data.get('confidence', 0.95):.2f})",
                'label': node_data['label'],
                'level': level,
                'chunks': node_data.get('chunks', []),
                'chunk_count': chunk_count,
                'chunk_details': [chunk_lookup.get(cid, {}) for cid in node_data.get('chunks', [])]
            },
            node_type=node_data.get('node_type', 'default'),
            source_position='right',
            target_position='left',
            style={
                'background': color,
                'border': '2px solid #333',
                'color': 'white',
                'border-radius': '10px',
                'padding': '10px',
                'font-size': '11px',
                'width': f'{size_factor}px',
                'height': '60px'
            },
            draggable=True,
            connectable=True
        )
        nodes.append(node)
    
    for edge_data in hierarchy.get('edges', []):
        edge = StreamlitFlowEdge(
            id=edge_data['id'],
            source=edge_data['source'],
            target=edge_data['target'],
            animated=True,
            style={'stroke': '#333', 'stroke-width': 2},
            marker_end={'type': 'arrowclosed'}
        )
        edges.append(edge)
    
    return nodes, edges

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏷️ Semantic Document Tagger</h1>
        <p>Intelligent document analysis with hierarchical tag visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Sidebar for input
    with st.sidebar:
        st.markdown("### 📄 Document Input")
        
        input_type = st.radio(
            "Choose input method:",
            ["📝 Paste Text", "📁 Upload PDF"],
            horizontal=True
        )
        
        text_content = ""
        
        if input_type == "📝 Paste Text":
            text_content = st.text_area(
                "Paste your text here:",
                height=300,
                placeholder="Enter your document text here..."
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload PDF file",
                type=['pdf'],
                help="Upload a PDF file to extract and analyze text"
            )
            
            if uploaded_file is not None:
                with st.spinner("Extracting text from PDF..."):
                    text_content = processor.extract_text_from_pdf(uploaded_file)
                
                if text_content:
                    st.success(f"✅ Extracted {len(text_content)} characters")
                    with st.expander("Preview extracted text"):
                        st.text(text_content[:500] + "..." if len(text_content) > 500 else text_content)
        
        # Processing settings
        st.markdown("### ⚙️ Processing Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            buffer_size = st.selectbox("Buffer Size", [1, 2, 3], index=0)
        with col2:
            threshold = st.selectbox("Threshold", [90, 95, 99], index=1)
        
        process_button = st.button(
            "🚀 Process Document",
            disabled=not text_content,
            use_container_width=True
        )
    
    # Main content area
    if text_content and process_button:
        # Check cache first
        file_hash = processor.get_file_hash(text_content)
        cached_result = processor.load_from_cache(file_hash)
        
        if cached_result:
            st.markdown('<div class="success-message">📱 Using cached results (document previously processed)</div>', unsafe_allow_html=True)
            chunks = cached_result['chunks']
            chunk_tags = cached_result['chunk_tags']
            hierarchy = cached_result['hierarchy']
        else:
            # Process document
            progress_container = st.container()
            
            with progress_container:
                st.markdown("### 🔄 Processing Pipeline")
                
                # Step 1: Semantic Chunking
                with st.spinner("🧩 Creating semantic chunks..."):
                    chunks = processor.semantic_chunk_text(text_content)
                
                if chunks:
                    st.success(f"✅ Created {len(chunks)} semantic chunks")
                    
                    # Step 2: Parallel Tag Generation
                    st.markdown("### 🏷️ Generating Tags")
                    st.info(f"Processing {len(chunks)} chunks in parallel...")
                    
                    chunk_tags = processor.generate_tags_parallel(chunks)
                    
                    if chunk_tags:
                        total_tags = sum(len(tags) for tags in chunk_tags.values())
                        st.success(f"✅ Generated {total_tags} tags across all chunks")
                        
                        # Step 3: Create Hierarchy
                        with st.spinner("🌳 Creating tag hierarchy..."):
                            hierarchy = processor.create_tag_hierarchy(chunk_tags, chunks)
                        
                        if hierarchy:
                            st.success("✅ Tag hierarchy created successfully")
                            
                            # Cache the results
                            cache_data = {
                                'chunks': chunks,
                                'chunk_tags': chunk_tags,
                                'hierarchy': hierarchy,
                                'processed_at': time.time()
                            }
                            processor.save_to_cache(file_hash, cache_data)
                        else:
                            st.error("Failed to create tag hierarchy")
                            return
                    else:
                        st.error("Failed to generate tags")
                        return
                else:
                    st.error("Failed to create semantic chunks")
                    return
        
        # Display Results
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(chunks)}</h3>
                <p>Semantic Chunks</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_tags = sum(len(tags) for tags in chunk_tags.values())
            st.markdown(f"""
            <div class="metric-card">
                <h3>{total_tags}</h3>
                <p>Total Tags</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            unique_tags = len(set().union(*chunk_tags.values()))
            st.markdown(f"""
            <div class="metric-card">
                <h3>{unique_tags}</h3>
                <p>Unique Tags</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            hierarchy_levels = len(set(node.get('level', 0) for node in hierarchy.get('nodes', [])))
            st.markdown(f"""
            <div class="metric-card">
                <h3>{hierarchy_levels}</h3>
                <p>Hierarchy Levels</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["🌳 Hierarchy Visualization", "📑 Chunks & Tags", "📈 Analysis Summary"])
        
        with tab1:
            st.markdown("### 🌳 Interactive Tag Hierarchy")
            
            if hierarchy and hierarchy.get('nodes'):
                # Convert to Streamlit Flow format
                nodes, edges = convert_to_streamlit_flow(hierarchy, chunks)
                
                # Initialize flow state with proper key parameter
                if 'flow_state' not in st.session_state or len(st.session_state.flow_state.nodes) != len(nodes):
                    st.session_state.flow_state = StreamlitFlowState(nodes=nodes, edges=edges)
                
                # Display flow diagram with updated state management
                st.session_state.flow_state = streamlit_flow(
                    'tag_hierarchy_flow',
                    st.session_state.flow_state,
                    layout='tree',
                    fit_view=True,
                    show_controls=True,
                    show_minimap=True,
                    height=600
                )
                
                # Handle node interactions through state
                if hasattr(st.session_state.flow_state, 'selected_id') and st.session_state.flow_state.selected_id:
                    selected_node = None
                    for node in st.session_state.flow_state.nodes:
                        if node.id == st.session_state.flow_state.selected_id:
                            selected_node = node
                            break
                    
                    if selected_node:
                        st.markdown("### 🔍 Selected Node Details")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Tag:** {selected_node.data.get('label', 'Unknown')}")
                            st.markdown(f"**Level:** {selected_node.data.get('level', 0)}")
                            st.markdown(f"**Associated Chunks:** {selected_node.data.get('chunk_count', 0)}")
                            
                            chunk_details = selected_node.data.get('chunk_details', [])
                            if chunk_details:
                                st.markdown("**Chunk Previews:**")
                                for chunk in chunk_details[:3]:  # Show first 3 chunks
                                    if chunk and 'text' in chunk:
                                        preview = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                                        st.markdown(f"""
                                        <div class="chunk-preview">
                                            <strong>{chunk.get('id', 'Unknown')}:</strong> {preview}
                                        </div>
                                        """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("**Actions:**")
                            if st.button("📋 Copy Node Info"):
                                st.success("Node info copied!")
                            
                            if st.button("🔗 Show Related"):
                                st.info("Feature coming soon!")
                else:
                    st.info("💡 Click on any node to see detailed information about that tag and its associated chunks.")
            else:
                st.warning("No hierarchy data available to visualize")
        
        with tab2:
            st.markdown("### 📑 Chunks and Generated Tags")
            
            for i, chunk in enumerate(chunks):
                with st.expander(f"Chunk {i+1} ({len(chunk['text'])} characters)"):
                    st.markdown(f"**Text Preview:**")
                    st.markdown(f"""
                    <div class="chunk-preview">
                        {chunk['text'][:300]}{'...' if len(chunk['text']) > 300 else ''}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("**Generated Tags:**")
                    tags = chunk_tags.get(chunk['id'], [])
                    tags_html = ''.join([f'<span class="tag-badge">{tag}</span>' for tag in tags])
                    st.markdown(tags_html, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### 📈 Analysis Summary")
            
            # Tag frequency analysis
            tag_frequency = {}
            for tags in chunk_tags.values():
                for tag in tags:
                    tag_frequency[tag] = tag_frequency.get(tag, 0) + 1
            
            # Sort by frequency
            sorted_tags = sorted(tag_frequency.items(), key=lambda x: x[1], reverse=True)
            
            st.markdown("**Most Frequent Tags:**")
            for tag, freq in sorted_tags[:10]:
                st.markdown(f"- **{tag}**: {freq} occurrences")
            
            # Document insights
            st.markdown("**Document Insights:**")
            st.markdown(f"""
            <div class="info-box">
                📄 <strong>Document Length:</strong> {len(text_content):,} characters<br>
                🧩 <strong>Average Chunk Size:</strong> {len(text_content) // len(chunks):,} characters<br>
                🏷️ <strong>Tags per Chunk:</strong> {total_tags / len(chunks):.1f} average<br>
                🎯 <strong>Tag Diversity:</strong> {unique_tags / total_tags * 100:.1f}% unique tags
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
