import streamlit as st
from llama_index.core import Settings, SimpleDirectoryReader, Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import tempfile

# --- UI CONFIGURATION ---
st.set_page_config(layout="wide")
st.title("📄 Advanced Semantic Chunker (Auto-Cache Method)")
st.markdown("""
This tool uses the same local model detection method as your RAG app. 
It relies on the `sentence-transformers` library's automatic caching.
""")

# --- MODEL CONFIGURATION ---
# We define the model name here, just like in your RAG application.
# The library will download this on the first run and use the local cache thereafter.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🔧 Controls")
breakpoint_percentile_threshold = st.sidebar.slider(
    "Breakpoint Percentile Threshold",
    min_value=80,
    max_value=99,
    value=95,
    help="Controls chunking sensitivity. Lower = more chunks, Higher = fewer chunks."
)

# --- MODEL LOADING & LLAMAINDEX SETTINGS ---
@st.cache_resource
def configure_llama_index():
    """
    Loads the embedding model using its Hugging Face identifier.
    The library handles caching automatically for offline use on subsequent runs.
    """
    st.write(f"Loading embedding model: '{EMBEDDING_MODEL_NAME}'")
    st.info("Note: The first run will download the model. Future runs will load from local cache.")
    
    try:
        # This line automatically handles downloading and caching.
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
        
        # Configure LlamaIndex to use this embedding model globally
        Settings.embed_model = embed_model
        st.success("Model loaded and configured successfully!")
        return True
    except Exception as e:
        st.error(f"Failed to load or download model. Check your internet connection for the first run. Error: {e}")
        return False

model_ready = configure_llama_index()


# --- CORE PROCESSING FUNCTION ---
def process_and_display_chunks(text_content):
    """Takes text, performs chunking, and displays the results."""
    st.success("Text loaded! Now performing semantic chunking...")
    
    documents = [Document(text=text_content)]

    # Instantiate the SemanticSplitterNodeParser. It will use the model from global Settings.
    parser = SemanticSplitterNodeParser(
        breakpoint_percentile_threshold=breakpoint_percentile_threshold
    )
    
    nodes = parser.get_nodes_from_documents(documents)

    st.header(f"Found {len(nodes)} Semantic Chunks")
    st.markdown(f"_(Based on a **{breakpoint_percentile_threshold}th percentile** similarity threshold)_")

    for i, node in enumerate(nodes):
        with st.expander(f"**Chunk {i + 1}** - ({len(node.get_content().split())} words)"):
            st.write(node.get_content())


# --- MAIN UI TABS ---
if model_ready:
    tab1, tab2 = st.tabs(["📤 Upload PDF", "📋 Paste Text"])
    
    with tab1:
        st.header("Option 1: Upload a PDF Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")
        
        if uploaded_file is not None:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    reader = SimpleDirectoryReader(input_dir=temp_dir)
                    docs_from_pdf = reader.load_data()
                    pdf_text = "".join([doc.get_content() for doc in docs_from_pdf])

                    if pdf_text.strip():
                        process_and_display_chunks(pdf_text)
                    else:
                        st.error("Could not extract any text from the PDF.")
            except Exception as e:
                st.error(f"An error occurred while processing the PDF: {e}")

    with tab2:
        st.header("Option 2: Paste Your Text")
        pasted_text = st.text_area("Paste your text here...", height=300, label_visibility="collapsed")
        
        if st.button("Chunk Pasted Text"):
            if pasted_text.strip():
                process_and_display_chunks(pasted_text)
            else:
                st.warning("Please paste some text into the text area.")
else:
    st.warning("Model could not be loaded. Please check the console for errors.")
