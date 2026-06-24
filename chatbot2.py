import streamlit as st
import os
import sqlite3
from groq import Groq
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PIL import Image
import pytesseract
import time

# =========================
# INIT
# =========================
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="AI Chatbot", layout="wide")

# =========================
# DB (Memory + Chats)
# =========================
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS memory(
    key TEXT PRIMARY KEY,
    value TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS chats(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT,
    message TEXT,
    chat_id TEXT
)
""")
conn.commit()

# =========================
# SESSION INIT
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(int(time.time()))

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# =========================
# MODEL (Embeddings)
# =========================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()
# =========================
# SAFETY FILTER
# =========================
bad_words = ["abuse", "idiot", "stupid"]

def safe_check(text):
    return not any(w in text.lower() for w in bad_words)

# =========================
# MEMORY
# =========================
def save_memory(key, value):
    cur.execute("INSERT OR REPLACE INTO memory VALUES (?,?)", (key, value))
    conn.commit()

def get_memory(key):
    cur.execute("SELECT value FROM memory WHERE key=?", (key,))
    row = cur.fetchone()
    return row[0] if row else None

# =========================
# PDF PROCESSING
# =========================
def process_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    vectors = embedder.encode(chunks)

    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors))

    st.session_state.vector_store = (index, chunks, vectors)

# =========================
# PDF SEARCH
# =========================
def search_pdf(query):
    if not st.session_state.vector_store:
        return ""

    index, chunks, vectors = st.session_state.vector_store

    q_vec = embedder.encode([query])
    D, I = index.search(np.array(q_vec), k=3)

    results = [chunks[i] for i in I[0]]
    return "\n".join(results)

# =========================
# GROQ RESPONSE
# =========================
def ask_groq(prompt, context=""):
    name = get_memory("name")
#system prompt 
    system= '''You are an AI assistant with memory and document understanding.

You must:
1. Use user name if known.
2. Prefer PDF context when available.
3. If answer is in context, quote it and explain simply.
4. If not in context, say: "Not found in document, answering from general knowledge."
5. Always stay accurate.
6. Never hallucinate PDF content.
7. Keep answers structured and easy to rea.'''

    if name:
        system += f" User name is {name}."

    messages = [
        {"role": "system", "content": system},
    ]

    if context:
        messages.append({"role": "system", "content": f"Context:\n{context}"})

    for m in st.session_state.messages[-6:]:
        messages.append(m)

    messages.append({"role": "user", "content": prompt})

    res = client.chat.completions.create(
       model="llama-3.1-8b-instant",
        messages=messages
    )

    return res.choices[0].message.content

# =========================
# GREETING
# =========================
def greeting(text):
    g = ["hi", "hello", "hey", "how are you"]
    if text.lower() in g:
        return "Hello 👋 How can I help you today?"
    return None

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🧠 AI Chatbot")

if st.sidebar.button("➕ New Chat"):
    st.session_state.messages = []
    st.session_state.chat_id = str(int(time.time()))

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.messages = []
    # =========================
# CHAT HISTORY
# =========================
st.sidebar.subheader("📜 Chat History")

cur.execute("""
SELECT chat_id,
       MIN(message)
FROM chats
WHERE role='user'
GROUP BY chat_id
ORDER BY chat_id DESC
""")

history = cur.fetchall()

for cid, first_msg in history:

    title = first_msg[:25] + "..." if len(first_msg) > 25 else first_msg

    if st.sidebar.button(title, key=f"history_{cid}"):

        cur.execute(
            "SELECT role,message FROM chats WHERE chat_id=? ORDER BY id",
            (cid,)
        )

        rows = cur.fetchall()

        st.session_state.messages = [
            {"role": role, "content": message}
            for role, message in rows
        ]

        st.session_state.chat_id = cid

        st.rerun()

# Download chat
chat_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])

st.sidebar.download_button(
    "⬇ Download Chat",
    chat_text,
    file_name="chat.txt"
)

# PDF Upload
pdf = st.sidebar.file_uploader("📄 Upload PDF", type=["pdf"])
if pdf:
    process_pdf(pdf)
    st.sidebar.success("PDF loaded!")

# Show memory
if get_memory("name"):
    st.sidebar.info(f"👤 Name: {get_memory('name')}")

# =========================
# MAIN UI
# =========================
st.title("💬 Hi how can i help you")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Type message...")

if user_input:

    if not safe_check(user_input):
        st.warning("⚠ Inappropriate message detected.")
        st.stop()

    # save name memory
    if "i'm " in user_input.lower():
        name = user_input.split("i'm")[-1].strip()
        save_memory("name", name)

    # greeting check
    greet = greeting(user_input)
    if greet:
        reply = greet
    else:
        pdf_context = search_pdf(user_input)
        reply = ask_groq(user_input, pdf_context)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": reply})

    # save to db
    cur.execute("INSERT INTO chats(role,message,chat_id) VALUES (?,?,?)",
                ("user", user_input, st.session_state.chat_id))
    cur.execute("INSERT INTO chats(role,message,chat_id) VALUES (?,?,?)",
                ("assistant", reply, st.session_state.chat_id))
    conn.commit()

    st.rerun()