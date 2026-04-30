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

/* GLOBAL */
body {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: white;
}
.block-container {
    padding-top: 2rem;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* HEADER */
.brand {
    text-align: center;
    margin-bottom: 25px;
}
.brand h1 {
    font-size: 44px;
    font-weight: 700;
}
.brand p {
    color: #94a3b8;
}

/* CHAT WRAPPER */
.chat-container {
    max-width: 900px;
    margin: auto;
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(18px);
    border-radius: 18px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.05);
    box-shadow: 0 0 40px rgba(0,0,0,0.6);
}

/* MESSAGES */
.user-msg {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    padding: 12px 16px;
    border-radius: 16px;
    margin: 10px 0;
    text-align: right;
}
.bot-msg {
    background: rgba(30, 41, 59, 0.6);
    padding: 14px 18px;
    border-radius: 16px;
    margin: 10px 0;
    border: 1px solid rgba(255,255,255,0.05);
}

/* INPUT */
.stTextInput input {
    border-radius: 999px;
    padding: 14px;
    background: rgba(15,23,42,0.6);
    border: 1px solid rgba(255,255,255,0.08);
}

/* UPLOADER */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 15px;
}

/* FOOTER */
.footer {
    text-align: center;
    margin-top: 40px;
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
    return SentenceTransformer("all-MiniLM-L6-v2")

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

# ---------------- UPLOAD ----------------
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

# ---------------- CHAT ----------------
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("💬 Ask your documents...")

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

# ---------------- DISPLAY ----------------
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for item in reversed(st.session_state.history):
    st.markdown(f"<div class='user-msg'>{item['q']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-msg'>{item['a']}</div>", unsafe_allow_html=True)

    with st.expander("🔎 Sources"):
        for txt in item["ctx"]:
            st.write(txt[:500] + "...")
        st.caption("📄 " + ", ".join(item["src"]))

    st.download_button("⬇️ Download", item["a"], file_name="answer.txt")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class='footer'>
NeuroVault AI • Crafted by <b>Anisha Chowdhury</b>
</div>
""", unsafe_allow_html=True)
