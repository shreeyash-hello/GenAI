import streamlit as st
from io import BytesIO
from typing import List, Tuple
import re
import requests
from pypdf import PdfReader

# ---------- OLLAMA CONFIG ----------

OLLAMA_BASE_URL = "http://localhost:11434"
CHAT_MODEL = "llama3.2"  

# ---------- PDF HANDLING ----------

def extract_text_from_pdf(file: BytesIO) -> str:
    """Read a PDF file (BytesIO) and return full text."""
    reader = PdfReader(file)
    pages_text: List[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)
        except Exception:
            continue
    return "\n\n".join(pages_text)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """Simple character-based chunking with overlap."""
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  

    return chunks


# ---------- SIMPLE KEYWORD RETRIEVAL (NO EMBEDDINGS) ----------

def tokenize(text: str) -> List[str]:
    """Very simple tokenizer: lowercase, alphanumeric words, filter very short tokens."""
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if len(t) > 2]


def build_index(chunks: List[str]) -> List[Tuple[str, set]]:
    """
    Build a simple index: for each chunk keep (chunk_text, set_of_tokens).
    """
    index: List[Tuple[str, set]] = []
    for ch in chunks:
        tokens = set(tokenize(ch))
        index.append((ch, tokens))
    return index


def score_chunk(question_tokens: List[str], chunk_tokens: set) -> float:
    """
    Score chunk with simple overlap:
    score = |intersection(question, chunk)| / (len(question_tokens) + 1)
    """
    if not question_tokens:
        return 0.0
    q_set = set(question_tokens)
    overlap = len(q_set & chunk_tokens)
    return overlap / (len(q_set) + 1.0)


def retrieve_relevant_chunks(
    question: str,
    indexed_chunks: List[Tuple[str, set]],
    top_k: int = 4,
) -> List[Tuple[str, float]]:
    """
    Rank chunks by simple keyword overlap and return top_k.
    """
    q_tokens = tokenize(question)
    scored = []

    for chunk_text, chunk_tokens in indexed_chunks:
        score = score_chunk(q_tokens, chunk_tokens)
        scored.append((chunk_text, score))

    # Sort by descending score
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]


# ---------- CHAT WITH OLLAMA ----------

def ask_ollama(context: str, question: str) -> str:
    """
    Use Ollama /api/chat with llama3.2 to answer based on context.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant answering questions about a PDF. "
                    "Use ONLY the provided context. "
                    "If the answer is not in the context, say you don't know."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context from the document:\n\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer clearly and concisely."
                ),
            },
        ],
        "stream": False,
    }

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # /api/chat returns {"message": {"role": "...", "content": "..."} ...}
    return data["message"]["content"]


# ---------- STREAMLIT APP ----------

st.set_page_config(
    page_title="Local PDF Q&A (Ollama - no embeddings)",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Local PDF Q&A App (Ollama)")
st.write(
    "Upload a PDF and ask questions about its content. "
    "This version uses simple keyword search (no embeddings) and Ollama's /api/chat."
)

# Session state
if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = None
if "raw_text" not in st.session_state:
    st.session_state.raw_text = None

with st.sidebar:
    st.header("How to use")
    st.markdown(
        """
        1. Upload a **PDF**.  
        2. Click **Process PDF** to index it.  
        3. Ask a question and click **Ask**.  

        Under the hood:
        - Extracts text from the PDF  
        - Splits into overlapping chunks  
        - Indexes chunks by simple keyword sets  
        - Picks best-matching chunks by word overlap  
        - Sends those chunks + your question to `llama3.2` via /api/chat  
        """
    )

uploaded_pdf = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"],
    help="The content is processed locally.",
)

process_clicked = st.button("📚 Process PDF", type="primary")

if uploaded_pdf and process_clicked:
    with st.spinner("Reading and indexing the PDF..."):
        try:
            pdf_text = extract_text_from_pdf(uploaded_pdf)
            if not pdf_text.strip():
                st.error("Could not extract text. Is this a scanned/image-only PDF?")
            else:
                chunks = chunk_text(pdf_text, chunk_size=800, overlap=200)
                if not chunks:
                    st.error("No chunks created from the PDF text.")
                else:
                    indexed = build_index(chunks)
                    st.session_state.indexed_chunks = indexed
                    st.session_state.raw_text = pdf_text
                    st.success(f"PDF processed! Created and indexed {len(chunks)} chunks.")
        except Exception as e:
            st.error(f"Error during processing: {e}")

st.markdown("---")

question = st.text_input(
    "Ask a question about the uploaded PDF",
    placeholder="e.g., What is the main conclusion of this document?",
)

ask_button = st.button("🤖 Ask")

if ask_button:
    if not st.session_state.indexed_chunks:
        st.warning("Please upload and process a PDF first.")
    elif not question.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                # Retrieve top chunks
                top_chunks = retrieve_relevant_chunks(
                    question,
                    st.session_state.indexed_chunks,
                    top_k=4,
                )

                # Build context (ignore very low-score chunks)
                non_zero = [(c, s) for c, s in top_chunks if s > 0]
                if not non_zero:
                    # Still send something to the model but warn it's likely not in the doc
                    context_text = ""
                else:
                    context_text = "\n\n---\n\n".join([c for c, _ in non_zero])

                answer = ask_ollama(context_text, question)

                st.markdown("### Answer")
                st.write(answer)

                with st.expander("Show retrieved chunks"):
                    for i, (chunk, score) in enumerate(top_chunks, start=1):
                        st.markdown(f"**Chunk {i} (score: {score:.3f})**")
                        st.write(chunk)
                        st.markdown("---")

            except Exception as e:
                st.error(f"Something went wrong while generating the answer: {e}")
