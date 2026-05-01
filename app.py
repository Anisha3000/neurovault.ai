import os
import uuid
import streamlit as st
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

# ---------------- CONFIG ----------------
st.set_page_config(page_title="NeuroVault AI", layout="wide")
api_key = st.secrets["GROQ_API_KEY"]

# ---------------- SESSION ----------------
if "chats" not in st.session_state:
    st.session_state.chats = {"Chat 1": []}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if "processing" not in st.session_state:
    st.session_state.processing = False

# ---------------- SIDEBAR ----------------
st.sidebar.title("💬 Chats")

for name in st.session_state.chats:
    if st.sidebar.button(name):
        st.session_state.current_chat = name

if st.sidebar.button("➕ New Chat"):
    new = f"Chat {len(st.session_state.chats)+1}"
    st.session_state.chats[new] = []
    st.session_state.current_chat = new
    st.rerun()

st.sidebar.markdown("---")

mode = st.sidebar.selectbox("Answer Style", ["Concise", "Detailed", "Bullet"])

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.chats[st.session_state.current_chat] = []
    st.rerun()

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center;'>NeuroVault AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>✦ ANISHA CHOWDHURY ✦</p>", unsafe_allow_html=True)

st.markdown("💡 Ask anything OR upload PDFs")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- GROQ ----------------
def ask_groq_stream(prompt):
    client = Groq(api_key=api_key)

    stream = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        stream=True
    )

    response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content
            yield response

# ---------------- PDF ----------------
def extract_text(file):
    reader = PdfReader(file)
    return " ".join([p.extract_text() or "" for p in reader.pages])

def chunk_text(text, size=200):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# ---------------- FILE ----------------
files = st.file_uploader("📄 Upload PDFs", type="pdf", accept_multiple_files=True)

if files:
    with st.spinner("Processing PDFs..."):
        chunks = []
        for f in files:
            chunks.extend(chunk_text(extract_text(f)))

        emb = model.encode(chunks)
        index = faiss.IndexFlatL2(emb.shape[1])
        index.add(np.array(emb))

        st.session_state.index = index
        st.session_state.chunks = chunks

    st.success("Documents ready")

# ---------------- CHAT ----------------
chat = st.session_state.chats[st.session_state.current_chat]

for msg in chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- INPUT ----------------
col1, col2 = st.columns([10,1])

with col1:
    user_input = st.text_input(
        "",
        value=st.session_state.input_text,
        placeholder="Ask anything...",
        key="input_box"
    )

with col2:
    send = st.button("➤")

# ---------------- SAFE SEND ----------------
def run_query(prompt):
    chat.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # context
    if st.session_state.index:
        q_vec = model.encode([prompt])
        D, I = st.session_state.index.search(np.array(q_vec), k=5)
        context = "\n\n".join([st.session_state.chunks[i] for i in I[0]])
    else:
        context = "General knowledge."

    style = {
        "Concise": "Be brief.",
        "Detailed": "Explain clearly.",
        "Bullet": "Use bullet points."
    }[mode]

    final_prompt = f"""
You are a smart AI assistant.

{style}

Context:
{context}

Question:
{prompt}
"""

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""

        for chunk in ask_groq_stream(final_prompt):
            full = chunk
            placeholder.markdown(full + "▌")

        placeholder.markdown(full)

    chat.append({"role": "assistant", "content": full})

# prevent double trigger
if send and user_input.strip() and not st.session_state.processing:
    st.session_state.processing = True

    run_query(user_input.strip())

    st.session_state.input_text = ""
    st.session_state.processing = False
    st.rerun()
