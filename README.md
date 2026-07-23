# 🚗 D.R.I.V.E.R.
**Document Retrieval & Intelligent Virtual Executive Researcher**

[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1.3+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-000000?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

D.R.I.V.E.R. is an advanced, production-grade executive AI assistant built on **Streamlit**, orchestrated via **LangGraph**, and powered by **Google Gemini**. It bridges private domain knowledge stored in Google Drive with real-time external intelligence retrieved from the web, featuring automated intent routing, user-tier quotas, and a live web OAuth authentication flow.

---

## 🌟 Key Features

* **🧠 AI Intent Router:** Automatically analyzes incoming prompts to classify tasks (`web`, `drive`, `hybrid`, or `general`) and determine complexity, loading only the necessary tools on the fly.
* **🚀 Daily Pro Quota & Tiering:** Automatically routes heavy tasks to high-power models (`gemini-1.5-pro`) while tracking daily usage via SQLite (capped at 5 Pro uses/day), gracefully falling back to efficient Flash models.
* **🌐 Google Web OAuth Integration:** Replaces legacy desktop credentials with a browser-based multi-user sign-in flow that securely retrieves profile data and session-based Drive tokens.
* **⚡ Live Token Streaming:** Features real-time typewriter effect response generation (`st.write_stream`) alongside interactive status updates for active tool execution.
* **💾 Persistent Conversation Memory:** SQLite-backed state retention via `langgraph-checkpoint-sqlite`. Conversations survive app restarts and remain partitioned per session using unique thread IDs.
* **🔍 Transparency & Audit Panel:** An interactive "Tool activity" dropdown that displays exact agent execution pathways, query arguments, and raw data responses in real-time.
* **⚙️ Custom Settings Modal:** Houses technical specifications, orchestration details, and developer credits (**B.Rakshan** / **RXN Studios**).

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
               │        Intent Router          │
               │   (Classifies Task & Tier)    │
               └──────────────┬────────────────┘
                              │
               ┌──────────────▼────────────────┐
               │   LangGraph ReAct Agent Core  │
               │  (Dynamic Gemini Model)       │
               └──────┬─────────────────┬──────┘
                      │                 │
     ┌────────────────▼───┐         ┌───▼────────────────┐
     │ Google Drive Search│         │ Tavily Web Search  │
     │  (Live User Token) │         │  (External Topic)  │
     └────────────────────┘         └────────────────────┘
                              │
               ┌──────────────▼────────────────┐
               │  SQLite Persistence Checkpoints│
               │  & Quota Tracker Database     │
               └───────────────────────────────┘
