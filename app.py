import os
import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from trulens_eval import instrument, TruApp
from trulens_eval.otel.semconv.trace import SpanAttributes
from trulens_eval.connectors.snowflake import SnowflakeConnector
from snowflake.cortex import complete
from trulens_eval.streamlit import trulens_feedback, trulens_trace, trulens_leaderboard

APP_NAME = "LocalPDFRAG"
APP_VERSION = "v1"

os.environ["TRULENS_OTEL_TRACING"] = "1"

def load_pdf_chunks(pdf_file, chunk_size=500):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    embedder = SentenceTransformer(model_name)
    chunk_embeddings = embedder.encode(chunks)
    chunk_embeddings = np.array(chunk_embeddings).astype("float32")
    return embedder, chunk_embeddings

def build_faiss_index(chunk_embeddings):
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings)
    return index

class LocalRAG:
    def __init__(self, chunks, index, embedder):
        self.chunks = chunks
        self.index = index
        self.embedder = embedder

    @instrument(span_type=SpanAttributes.SpanType.RETRIEVAL)
    def retrieve_context(self, query, top_k=4):
        query_vec = self.embedder.encode([query]).astype("float32")
        D, I = self.index.search(query_vec, top_k)
        return [self.chunks[i] for i in I[0]]

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate_completion(self, query, context_str):
        prompt = f"""
You are an expert assistant extracting information from context provided.
Answer the question in long-form, fully and completely, based on the context. Do not hallucinate.
If you don´t have the information just say so. If you do have the information you need, just tell me the answer.

Context: {context_str}

Question:
{query}

Answer:
"""
        response = ""
        stream = complete("mistral-large2", prompt, stream=True)
        for update in stream:
            response += update
            yield update
        return response

    @instrument(span_type=SpanAttributes.SpanType.RECORD_ROOT)
    def query(self, query):
        context_list = self.retrieve_context(query)
        context_str = "\n\n".join(context_list)
        return context_str, self.generate_completion(query, context_str)

def create_snowflake_connector():
    sf_connector = SnowflakeConnector(
        user=os.getenv("SF_USER"),
        password=os.getenv("SF_PASSWORD"),
        account=os.getenv("SF_ACCOUNT"),
        warehouse=os.getenv("SF_WAREHOUSE"),
        database=os.getenv("SF_DATABASE"),
        schema=os.getenv("SF_SCHEMA")
    )
    return sf_connector

def register_tru_app(rag, connector):
    tru_app = TruApp(
        test_app=rag,
        app_name=APP_NAME,
        app_version=APP_VERSION,
        connector=connector,
        main_method=rag.query
    )
    return tru_app

def get_latest_app_record(app_name):
    from trulens_eval.tru_db import TruDB
    db = TruDB()
    records = db.get_records(app_id=app_name)
    if records:
        return records[-1]
    else:
        return None

def show_observability_info(record):
    st.header("Observability Dashboard")
    if record is not None:
        st.subheader("Feedback Scores")
        trulens_feedback(record=record)
        st.subheader("Execution Trace")
        trulens_trace(record=record)
    st.subheader("Leaderboard (App-wide Metrics)")
    trulens_leaderboard()
    st.markdown("""
---
**How to View in Snowsight:**
1. Open Snowsight and go to your AI Observability schema (e.g., `AI_OBSERVABILITY.EVENTS`).
2. Run: `SELECT * FROM AI_OBSERVABILITY.EVENTS WHERE APP_ID = '%s' ORDER BY EVENT_TIMESTAMP DESC LIMIT 100;`
3. Use Dashboards to visualize metrics like relevance, latency, and cost.
4. Drill into events for full traces and feedback.
""" % APP_NAME)

st.title("RAG + Snowflake AI Observability Demo")
st.markdown("Upload a PDF, ask questions, and observe real-time AI observability metrics.")

pdf_file = st.file_uploader("Upload your PDF", type="pdf")
if pdf_file:
    with st.spinner("Loading and embedding PDF..."):
        chunks = load_pdf_chunks(pdf_file)
        embedder, chunk_embeddings = embed_chunks(chunks)
        index = build_faiss_index(chunk_embeddings)
    rag = LocalRAG(chunks, index, embedder)
    sf_connector = create_snowflake_connector()
    tru_app = register_tru_app(rag, sf_connector)
    st.success("PDF loaded and indexed.")
    user_query = st.text_input("Ask a question about your PDF")
    if user_query:
        with st.spinner("Retrieving and generating answer..."):
            context_str, answer_stream = tru_app.query(user_query)
            st.subheader("Retrieved Context")
            st.code(context_str)
            st.subheader("Answer (from Snowflake LLM)")
            answer_placeholder = st.empty()
            answer_text = ""
            for chunk in answer_stream:
                answer_text += chunk
                answer_placeholder.write(answer_text)
            st.success("Done.")
        record = get_latest_app_record(APP_NAME)
        show_observability_info(record)
else:
    st.info("Please upload a PDF to get started.")

st.markdown("""
All observability features (tracing, metrics, evaluations) are available even without ground truth.
For advanced visualizations, open Snowsight and explore the EVENTS table or connect to your preferred observability dashboard.
""")
