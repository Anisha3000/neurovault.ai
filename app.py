import streamlit as st
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

# ---------------- CONFIG ----------------
st.set_page_config(page_title="NeuroVault AI", layout="wide")

# ---------------- SAFE API KEY ----------------
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    st.error("⚠️ API key missing. Contact developer.")
    st.stop()

# ---------------- PREMIUM UI ----------------
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: radial-gradient(circle at top, #020617, #020617);
    color: #e2e8f0;
}

/* TITLE */
h1 {
    text-align: center;
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(90deg,#38bdf8,#6366f1,#a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* CENTER CONTENT */
.block-container {
    max-width: 850px;
    padding-top: 2rem;
}

/* USER MESSAGE */
[data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"]) {
    border-radius: 16px;
}

/* ASSISTANT */
[data-testid="stChatMessage"] {
    background: rgba(30,41,59,0.6);
    padding: 12px;
    margin-bottom: 10px;
}

/* INPUT */
[data-testid="stChatInput"] textarea {
    border-radius: 12px !important;
    background: #020617;
    color: white;
}

/* SCROLL SMOOTH */
html {
    scroll-behavior: smooth;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []

# ---------------- TITLE ----------------
st.markdown("<h1>NeuroVault AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>✦ ANISHA CHOWDHURY ✦</p>", unsafe_allow_html=True)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- PDF ----------------
def extract_text(file):
    try:
        reader = PdfReader(file)
        return " ".join([p.extract_text() or "" for p in reader.pages])
    except:
        return ""

def chunk_text(text, size=150):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# ---------------- FILE ----------------
files = st.file_uploader("📄 Upload PDFs (optional)", type="pdf", accept_multiple_files=True)

if files:
    try:
        with st.spinner("Processing PDFs..."):
            chunks = []
            for f in files:
                chunks.extend(chunk_text(extract_text(f)))

            if chunks:
                emb = model.encode(chunks)
                index = faiss.IndexFlatL2(emb.shape[1])
                index.add(np.array(emb))

                st.session_state.index = index
                st.session_state.chunks = chunks

        st.success("✅ Documents ready")

    except Exception:
        st.warning("⚠️ Failed to process PDF")

# ---------------- GROQ (SAFE) ----------------
def ask_groq_stream(prompt):
    try:
        client = Groq(api_key=api_key)

        prompt = prompt[:3000]  # HARD LIMIT

        stream = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=800,
            stream=True
        )

        response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
                yield response

    except Exception:
        yield "⚠️ Server busy or input too large. Try again."

# ---------------- CHAT DISPLAY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- QUERY ----------------
def run_query(prompt):

    # ADD USER
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""

        try:
            # SAFE CONTEXT
            if st.session_state.index:
                q_vec = model.encode([prompt])
                D, I = st.session_state.index.search(np.array(q_vec), k=3)

                selected = [
                    st.session_state.chunks[i]
                    for i in I[0]
                    if i < len(st.session_state.chunks)
                ]

                context = "\n\n".join(selected)[:1200]

            else:
                context = "General knowledge."

            final_prompt = f"""
You are a premium AI assistant.

Give clear, smart, structured answers.

Context:
{context}

Question:
{prompt}
"""

            # STREAM
            for chunk in ask_groq_stream(final_prompt):
                full = chunk
                placeholder.markdown(full + "▌")

            placeholder.markdown(full)

        except Exception:
            full = "⚠️ Something went wrong. Try again."
            placeholder.markdown(full)

    st.session_state.messages.append({"role": "assistant", "content": full})

# ---------------- INPUT ----------------
user_input = st.chat_input("Ask anything...")

if user_input and user_input.strip():
    run_query(user_input.strip())
    st.rerun()
