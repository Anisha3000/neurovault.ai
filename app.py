import os
import streamlit as st
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# ---------------- CONFIG ----------------
st.set_page_config(page_title="NeuroVault AI", layout="wide")

# ---------------- PREMIUM UI ----------------
st.markdown("""
<style>

/* -------- GLOBAL -------- */
html, body {
    background: radial-gradient(circle at 20% 0%, #0f172a, #020617);
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
}

.block-container {
    max-width: 1100px;
    margin: auto;
    padding-top: 2rem;
}

/* -------- SIDEBAR -------- */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(25px);
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* -------- HEADER -------- */
.brand {
    text-align: center;
    margin-bottom: 25px;
}

.brand h1 {
    font-size: 54px;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.brand p {
    color: #94a3b8;
    font-size: 15px;
}

/* -------- BADGE -------- */
.badge {
    text-align: center;
    margin-bottom: 20px;
}

/* -------- MAIN CARD -------- */
.card {
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(25px);
    border-radius: 22px;
    padding: 28px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow:
        0 10px 40px rgba(0,0,0,0.8),
        inset 0 1px 0 rgba(255,255,255,0.04);
}

/* -------- UPLOADER -------- */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(255,255,255,0.15);
    border-radius: 18px;
    padding: 18px;
    background: rgba(255,255,255,0.02);
}

/* -------- INPUT -------- */
.stTextInput input {
    border-radius: 999px;
    padding: 16px;
    font-size: 15px;
    background: rgba(15,23,42,0.7);
    border: 1px solid rgba(255,255,255,0.08);
    transition: 0.3s;
}

.stTextInput input:focus {
    border: 1px solid #6366f1;
    box-shadow: 0 0 15px rgba(99,102,241,0.4);
}

/* -------- BUTTON -------- */
button[kind="primary"] {
    border-radius: 999px !important;
    background: linear-gradient(135deg, #6366f1, #2563eb);
    border: none;
    font-weight: 600;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4);
}

/* -------- CHAT -------- */
.user-msg {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    padding: 14px 18px;
    border-radius: 18px;
    margin: 12px 0;
    text-align: right;
    box-shadow: 0 6px 20px rgba(37,99,235,0.3);
}

.bot-msg {
    background: rgba(30, 41, 59, 0.7);
    padding: 16px 20px;
    border-radius: 18px;
    margin: 12px 0;
    border: 1px solid rgba(255,255,255,0.05);
}

/* -------- FOOTER -------- */
.footer {
    text-align: center;
    margin-top: 50px;
    color: #64748b;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class='brand'>
    <h1>🧠 NeuroVault AI</h1>
    <p>Private AI workspace for deep document intelligence</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='badge'>
    <span style='padding:6px 14px; border-radius:999px;
    background:rgba(99,102,241,0.2); color:#a5b4fc; font-size:12px;'>
    ⚡ AI + Vector Search Engine
    </span>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## 🚀 Workspace")

api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")

mode = st.sidebar.selectbox(
    "🧾 Answer Style",
    ["Concise", "Detailed", "Bullet points"]
)

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.history = []

if st.sidebar.button("♻️ Reset Documents"):
    st.session_state.clear()
    st.rerun()

client = OpenAI(api_key=api_key) if api_key else None

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

model = load_model()

# ---------------- HELPERS ----------------
def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for p in reader.pages:
        t = p.extract_text()
        if t:
            text += t
    return text

def chunk_text(text, size=200):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# ---------------- MAIN CARD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

# Upload
files = st.file_uploader("📄 Upload PDFs", type="pdf", accept_multiple_files=True)

if files and "index" not in st.session_state:
    with st.spinner("⚡ Processing documents..."):
        chunks, sources = [], []

        for f in files:
            text = extract_text(f)
            ch = chunk_text(text)
            chunks.extend(ch)
            sources.extend([f.name]*len(ch))

        emb = model.encode(chunks)

        index = faiss.IndexFlatL2(emb.shape[1])
        index.add(np.array(emb))

        st.session_state.index = index
        st.session_state.chunks = chunks
        st.session_state.sources = sources

    st.success("✅ Documents ready")

# Chat input
query = st.text_input("", placeholder="Ask anything about your documents...")

# Chat logic
if "history" not in st.session_state:
    st.session_state.history = []

if query and "index" in st.session_state:
    q_vec = model.encode([query])
    D, I = st.session_state.index.search(np.array(q_vec), k=3)

    ctx_chunks = [st.session_state.chunks[i] for i in I[0]]
    context = "\n\n".join(ctx_chunks)
    src = list(set([st.session_state.sources[i] for i in I[0]]))

    style_prompt = {
        "Concise": "Answer briefly.",
        "Detailed": "Give a detailed explanation.",
        "Bullet points": "Answer in bullet points."
    }[mode]

    answer = ""

    if client:
        with st.spinner("🤖 Thinking..."):
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                stream=True,
                messages=[
                    {"role":"system","content":f"Use ONLY context. {style_prompt}"},
                    {"role":"user","content":f"{context}\n\nQ:{query}"}
                ]
            )
            placeholder = st.empty()
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    answer += chunk.choices[0].delta.content
                    placeholder.markdown(answer)
    else:
        answer = context

    st.session_state.history.append({
        "q": query,
        "a": answer,
        "src": src,
        "ctx": ctx_chunks
    })

# Display chat
for item in reversed(st.session_state.history):
    st.markdown(f"<div class='user-msg'>{item['q']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-msg'>{item['a']}</div>", unsafe_allow_html=True)

    with st.expander("🔎 Sources"):
        for txt in item["ctx"]:
            st.write(txt[:500] + "...")
        st.caption("📄 " + ", ".join(item["src"]))

    st.download_button("⬇️ Download Answer", item["a"], file_name="answer.txt")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class='footer'>
NeuroVault AI • Crafted by <b>Anisha Chowdhury</b>
</div>
""", unsafe_allow_html=True)
