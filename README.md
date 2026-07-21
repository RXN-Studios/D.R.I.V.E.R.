# 🚗 D.R.I.V.E.R.
### Document Retrieval & Intelligent Virtual Executive Researcher

[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1.3+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-000000?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

D.R.I.V.E.R. is a single-file, production-hardened executive AI assistant built on **Streamlit** and orchestrated via **LangGraph**. Powered by **Google Gemini**, D.R.I.V.E.R. bridges the gap between private domain knowledge stored in Google Drive and external facts retrieved live from the web.

---

## 🌟 Key Features

* **🧠 Swappable Reasoning Engine:** Runs on Gemini models (`gemini-2.5-flash` by default) using LangChain's latest `create_agent` framework.
* **📂 Full-Text Google Drive Retrieval:** Directly searches, reads, and synthesizes content across Google Docs, PDFs, and spreadsheets with file-level attribution.
* **🌐 Live Web Research:** Uses Tavily Search to incorporate real-time external data and URL source citations.
* **💾 Persistent Conversation Memory:** SQLite-backed state retention via `langgraph-checkpoint-sqlite`. Conversations survive app restarts and remain partitioned per browser session using unique `thread_id` keys.
* **☁️ Headless Cloud Ready:** Designed for seamless deployment on **Streamlit Community Cloud** using pre-authorized OAuth token injection via `st.secrets`.
* **🔍 Source Attribution & Audit Panel:** Interactive "Tool Activity" dropdowns display exact agent execution pathways, query parameters, and raw tool responses in real-time.

---

## 🛠️ Architecture

```text
               ┌───────────────────────────────┐
               │    Streamlit Web Interface    │
               └──────────────┬────────────────┘
                              │
                      User Input / Messages
                              │
               ┌──────────────▼────────────────┐
               │   LangGraph Agent Core        │
               │  (Gemini Reasoning Engine)    │
               └──────┬─────────────────┬──────┘
                      │                 │
     ┌────────────────▼───┐         ┌───▼────────────────┐
     │ Google Drive Search│         │ Tavily Web Search  │
     │  (Internal Knowledge)        │  (External Research)│
     └────────────────────┘         └────────────────────┘
                              │
               ┌──────────────▼────────────────┐
               │  SQLite Persistence Saver     │
               │  (Thread-based Checkpointing) │
               └───────────────────────────────┘
