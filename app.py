import os
import uuid
import streamlit as st
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

# ---------------- MIC SAFE ----------------
try:
    from streamlit_mic_recorder import mic_recorder
    mic_enabled = True
except:
    mic_enabled = False

st.set_page_config(page_title="NeuroVault AI", layout="wide")

# ---------------- SESSION ----------------
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())[:8]

if "chats" not in st.session_state:
    st.session_state.chats = {"New Chat": []}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "New Chat"

if "run" not in st.session_state:
    st.session_state.run = False

if "query" not in st.session_state:
    st.session_state.query = ""

# ---------------- THEME ----------------
dark = st.sidebar.toggle("🌗 Dark mode", value=True)

bg = "#020617" if dark else "#f8fafc"
text = "#e2e8f0" if dark else "#111"
bot_bg = "rgba(30,41,59,0.9)" if dark else "#ffffff"
user_bg = "#2563eb" if dark else "#3b82f6"

# THEME REFRESH
if "prev_theme" not in st.session_state:
    st.session_state.prev_theme = dark

if st.session_state.prev_theme != dark:
    st.session_state.prev_theme = dark
    st.rerun()

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("### 💬 Conversations")

if st.sidebar.button("🔄 Refresh App"):
    st.rerun()

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.chats[st.session_state.current_chat] = []
    st.rerun()

for name in st.session_state.chats:
    if st.sidebar.button(name):
        st.session_state.current_chat = name

if st.sidebar.button("➕ New Chat"):
    new = f"Chat {len(st.session_state.chats)+1}"
    st.session_state.chats[new] = []
    st.session_state.current_chat = new

# 🔑 GROQ KEY
api_key = st.sidebar.text_input("🔑 Groq API Key", type="password")

mode = st.sidebar.selectbox("Answer style", ["Concise", "Detailed", "Bullet"])

# ---------------- UI ----------------
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-color: {bg} !important;
    color: {text} !important;
}}
[data-testid="stSidebar"] {{
    background-color: {"#020617" if dark else "#ffffff"} !important;
}}
.title {{
    font-size: 56px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg,#38bdf8,#6366f1,#a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.creator {{
    text-align: center;
    letter-spacing: 4px;
    background: linear-gradient(90deg,#22d3ee,#a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.user-msg {{
    background: {user_bg};
    padding: 14px;
    border-radius: 16px;
    text-align: right;
    margin: 10px 0;
    color: white;
}}
.bot-msg {{
    background: {bot_bg};
    padding: 16px;
    border-radius: 16px;
    margin: 10px 0;
}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='title'>NeuroVault AI</div>
<div class='creator'>✦ ANISHA CHOWDHURY ✦</div>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- GROQ ----------------
def ask_groq(prompt):
    if not api_key:
        return "⚠️ Enter Groq API Key in sidebar"

    try:
        client = Groq(api_key=api_key)

        res = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        return res.choices[0].message.content

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ---------------- PDF ----------------
def extract_text(file):
    try:
        reader = PdfReader(file)
        return " ".join([p.extract_text() or "" for p in reader.pages])
    except:
        return ""

def chunk_text(text, size=200):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# ---------------- FILE ----------------
files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if files and "index" not in st.session_state:
    with st.spinner("⚡ Processing..."):
        chunks = []
        for f in files:
            chunks.extend(chunk_text(extract_text(f)))

        emb = model.encode(chunks)
        index = faiss.IndexFlatL2(emb.shape[1])
        index.add(np.array(emb))

        st.session_state.index = index
        st.session_state.chunks = chunks

    st.success("✅ Documents ready")

# ---------------- INPUT ----------------
col1, col2, col3 = st.columns([6,1,1])

with col1:
    user_input = st.text_input("", value=st.session_state.query, placeholder="Ask anything...")

with col2:
    if st.button("▶️"):
        st.session_state.query = user_input
        st.session_state.run = True

with col3:
    if mic_enabled:
        mic_recorder(start_prompt="🎙", stop_prompt="Stop")

# ---------------- CHAT ----------------
chat = st.session_state.chats[st.session_state.current_chat]

if st.session_state.run:

    query = st.session_state.query

    with st.spinner("🤖 Thinking..."):

        if "index" in st.session_state:
            q_vec = model.encode([query])
            D, I = st.session_state.index.search(np.array(q_vec), k=5)
            context = "\n\n".join([st.session_state.chunks[i] for i in I[0]])
        else:
            context = "No documents provided."

        style = {
            "Concise": "Be brief.",
            "Detailed": "Explain clearly.",
            "Bullet": "Use bullet points."
        }[mode]

        prompt = f"""
You are a smart AI assistant.

{style}

Context:
{context}

Question:
{query}
"""

        full = ask_groq(prompt)

    chat.append({"q": query, "a": full})

    st.session_state.run = False
    st.session_state.query = ""
    st.rerun()

# ---------------- DISPLAY ----------------
for item in reversed(chat):
    st.markdown(f"<div class='user-msg'>{item['q']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-msg'>{item['a']}</div>", unsafe_allow_html=True)
