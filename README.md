AI Chatbot with Groq + PDF RAG + Memory

A smart AI chatbot built using Streamlit + Groq LLM + FAISS + SQLite.
It supports PDF question answering, chat memory, and conversation history.


Features
💬 Chat with AI (Groq LLM)
📄 Upload PDF and ask questions (RAG system)
🧠 Memory (remembers user name)
📜 Chat history saved (SQLite)
🔍 Semantic search using FAISS
⚡ Fast Streamlit UI

How it works
User sends message
If PDF uploaded → FAISS retrieves relevant text
Context + chat history sent to Groq
AI generates response
Stored in SQLite database

Future Improvements
🔐 User login system
☁ Cloud deployment
📊 Better UI dashboard
🧾 Multiple PDF support
⚡ Faster vector DB optimization