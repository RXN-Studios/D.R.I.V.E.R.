"""
D.R.I.V.E.R. — Document Retrieval & Intelligent Virtual Executive Researcher
=============================================================================

A single-file Streamlit chat app backed by a LangGraph-powered agent
(`langchain.agents.create_agent`) running Gemini as the reasoning engine,
with two tools:

  1. Web Search      -> Tavily (`langchain_tavily.TavilySearch`)
  2. Drive Retrieval -> Google Drive full-text search + document loading
                        (`langchain_google_community.GoogleDriveLoader`)

Run with:
    streamlit run app.py

-----------------------------------------------------------------------------
PRODUCTION-HARDENING NOTES (v2 — updated for commercial/long-term use)
-----------------------------------------------------------------------------
1. Agent construction now uses `langchain.agents.create_agent` instead of
   the now-deprecated `langgraph.prebuilt.create_react_agent`. Same
   `.invoke()` interface, no deprecation warning, and it's the path
   LangChain is actively investing in (tested against langgraph==1.2.9 /
   langchain==1.3.14).

2. Conversation memory now persists to a local SQLite file
   (`langgraph.checkpoint.sqlite.SqliteSaver`) instead of the in-memory
   checkpointer. The previous version lost every conversation on restart —
   fine for a demo, not for something you're shipping. SqliteSaver has an
   internal lock, so it's safe across the multiple threads one Streamlit
   process uses to serve concurrent sessions. If you later scale this out
   to multiple server processes/replicas, or need heavy concurrent write
   throughput, migrate to `langgraph-checkpoint-postgres`'s `PostgresSaver`
   — the swap is ~3 lines since the checkpointer is fully abstracted.

3. API keys now resolve from `st.secrets` first, then environment
   variables, and only fall back to an editable sidebar field if neither
   is set. This matters once this is a hosted, multi-visitor app: you
   don't want random visitors pasting (or reading) your Gemini/Tavily keys
   in a public sidebar. Configure `.streamlit/secrets.toml` (or your
   host's secret manager) with GOOGLE_API_KEY / TAVILY_API_KEY and the
   sidebar fields simply won't appear.

4. Google Drive OAuth still has one real constraint worth knowing:
   `InstalledAppFlow.run_local_server()` opens a *local* browser tab and a
   *local* port, which only works when this app runs on the same machine
   as the browser doing the consenting. That's fine for local/personal use
   (matching the original design doc's "localhost:8501" model) but it
   cannot complete on a headless remote server. This file now also accepts
   a pre-generated token via `st.secrets["GOOGLE_DRIVE_TOKEN_JSON"]`, so
   the one-time browser consent can happen once on your laptop, and the
   resulting token.json contents get pasted into the host's secrets for
   the deployed app to reuse — no browser needed on the server itself.
   This still assumes ONE shared Google identity/Drive for the whole
   deployed app (single-tenant). If instead each of your customers should
   connect their *own* separate Drive, that's a materially different (and
   bigger) build — a real per-user OAuth redirect flow with a token store
   — flagged separately, not silently done here.
"""

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from datetime import date
import streamlit as st
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_community import GoogleDriveLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver
import requests
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
# =============================================================================
# Constants
# =============================================================================

DEFAULT_MODEL = "gemini-2.5-flash"  # Swap freely — e.g. "gemini-3-flash-preview"
                                     # or "gemini-3.5-flash" if your key has access.
                                     # This is the "Swappable Reasoning Engine".

# Broad read scope so Drive search can find any file the user can see,
# not just files this app created/opened (the library's default `drive.file`
# scope is too narrow for a general search assistant).
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def get_secret(key: str, default: str = "") -> str:
    """
    Resolve a config value the "commercial deployment" way:
    `st.secrets` (Streamlit Cloud / most hosts' secret manager) first, then
    an environment variable, then a default. Accessing `st.secrets` raises
    if no secrets.toml exists at all, hence the try/except.
    """
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


SYSTEM_PROMPT = """You are D.R.I.V.E.R. (Document Retrieval & Intelligent Virtual \
Executive Researcher), an executive assistant with two tools:

1. `google_drive_search` — full-text searches the user's own Google Drive and \
returns matching document content. Use this for anything about the user's own \
files, notes, projects, or stored knowledge.
2. `tavily_search` — searches the live web. Use this for current events, external \
facts, or anything unlikely to live in the user's Drive.

Rules:
- Decide which tool(s) the question needs. If it requires both internal context \
and external facts, call BOTH tools before answering.
- Always cite sources. For Drive results, cite the exact file name/title from the \
tool output. For web results, cite the URL.
- If neither tool finds anything relevant, say so plainly — never invent an answer.
- Be concise and direct.
"""


# =============================================================================
# Google Drive: credentials + a real "search by query" tool
# =============================================================================

def build_drive_search_tool(creds_json_str: str, num_results: int):
    """
    Factory that returns a LangChain tool using live user session credentials.
    """
    @tool("google_drive_search")
    def google_drive_search(query: str) -> str:
        """Search the user's personal Google Drive for documents matching the
        query, and return their text content along with the file name/title."""
        try:
            from googleapiclient.discovery import build
            
            # Load the credentials directly from the user's session string
            creds = Credentials.from_authorized_user_info(json.loads(creds_json_str))
            drive_service = build("drive", "v3", credentials=creds)

            safe_query = query.replace("'", "\\'")
            response = (
                drive_service.files()
                .list(
                    q=f"fullText contains '{safe_query}' and trashed = false",
                    pageSize=num_results,
                    fields="files(id, name, mimeType)",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            matches = response.get("files", [])
            if not matches:
                return "No matching documents were found in Google Drive."

            loader = GoogleDriveLoader(
                file_ids=[f["id"] for f in matches],
                credentials=creds,
            )
            docs = loader.load()
        except Exception as exc:
            return f"Google Drive search failed: {exc}"

        if not docs:
            return "Matching files found, but no readable text could be extracted."

        chunks = []
        for doc in docs:
            title = doc.metadata.get("title", "Untitled")
            source = doc.metadata.get("source", "")
            content = doc.page_content.strip()[:4000]
            chunks.append(f"### {title}\nSource: {source}\n\n{content}")

        return "\n\n---\n\n".join(chunks)

    return google_drive_search
# =============================================================================
# Small helper: Gemini can return content as a plain string OR as a list of
# content blocks (e.g. [{"type": "text", "text": "..."}]) depending on the
# response. Normalize either shape to a plain string for display.
# =============================================================================

def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


# =============================================================================
# Persistent conversation memory (SQLite-backed checkpointer)
# =============================================================================
# `st.cache_resource` makes this a process-wide singleton: every browser
# session shares the SAME connection/file, and conversations stay isolated
# from each other via `thread_id` (set per-session below). This is exactly
# what you want — one durable store, partitioned per conversation — rather
# than one checkpointer per session, which would defeat the point of
# persisting anything.

@st.cache_resource
def get_checkpointer(db_path: str) -> SqliteSaver:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()  # idempotent — creates tables on first run only
    return checkpointer

# =============================================================================
# Intent Routing & Quota Tracking
# =============================================================================
class TaskIntent(BaseModel):
    task_type: str = Field(description="Must be one of: 'drive' (files/docs), 'web' (internet research), 'hybrid' (both), or 'general' (basic chat, no search needed)")
    complexity: str = Field(description="Must be one of: 'low' or 'high'")

def setup_quota_db(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS user_quotas (date TEXT PRIMARY KEY, pro_uses INTEGER)")
    conn.commit()
    return conn

def get_pro_usage(conn):
    today = date.today().isoformat()
    cur = conn.execute("SELECT pro_uses FROM user_quotas WHERE date = ?", (today,))
    row = cur.fetchone()
    return row[0] if row else 0

def increment_pro_usage(conn):
    today = date.today().isoformat()
    usage = get_pro_usage(conn)
    if usage == 0:
        conn.execute("INSERT INTO user_quotas (date, pro_uses) VALUES (?, 1)", (today,))
    else:
        conn.execute("UPDATE user_quotas SET pro_uses = ? WHERE date = ?", (usage + 1, today,))
    conn.commit()
    return usage + 1
  
# =============================================================================
# Chat History Tracking Helpers
# =============================================================================
def save_thread_title(db_path: str, thread_id: str, title: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS conversation_threads (thread_id TEXT PRIMARY KEY, title TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    cur = conn.execute("SELECT title FROM conversation_threads WHERE thread_id = ?", (thread_id,))
    if not cur.fetchone():
        conn.execute("INSERT INTO conversation_threads (thread_id, title) VALUES (?, ?)", (thread_id, title))
        conn.commit()
    conn.close()

def get_saved_threads(db_path: str):
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS conversation_threads (thread_id TEXT PRIMARY KEY, title TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        cur = conn.execute("SELECT thread_id, title FROM conversation_threads ORDER BY rowid DESC")
        return cur.fetchall()
    except Exception:
        return []
    finally:
        conn.close()

def load_messages_from_checkpoint(checkpointer, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint_tuple = checkpointer.get(config)
    if not checkpoint_tuple or not checkpoint_tuple.checkpoint:
        return []
    messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", [])
    ui_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            ui_messages.append({"role": "user", "content": extract_text(m.content)})
        elif isinstance(m, AIMessage):
            txt = extract_text(m.content)
            if txt.strip():
                ui_messages.append({"role": "assistant", "content": txt})
    return ui_messages
# =============================================================================
# Streamlit page setup
# =============================================================================

st.set_page_config(page_title="D.R.I.V.E.R.", page_icon="🚗", layout="wide")
st.title("🚗 D.R.I.V.E.R.")
st.caption("Data Retriever & Intelligent Virtual Executive Researcher")
import time

# Initialize the empty placeholder
splash = st.empty()

with splash.container():
    st.markdown(
        """
        <style>
        .splash-screen {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-color: #0e1117;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 999999;
            /* Added keyframe definition for fadeOut */
            animation: fadeOut 1s ease-in-out 1.5s forwards; 
        }
        .splash-text {
            font-size: 4rem;
            font-weight: bold;
            color: #D9DEE5; 
            text-align: center;
            animation: fadeInOut 2.5s ease-in-out forwards;
        }
        
        /* Keyframe definitions */
        @keyframes fadeOut {
            0% { opacity: 1; }
            100% { opacity: 0; }
        }
        @keyframes fadeInOut {
            0% { opacity: 0; transform: scale(0.9); }
            30% { opacity: 1; transform: scale(1); }
            80% { opacity: 1; transform: scale(1); }
            100% { opacity: 0; transform: scale(1.1); }
        }
        </style>
        
        <!-- Wrapped inside the splash-screen div to display fullscreen background -->
        <div class="splash-screen">
            <div class="splash-text">RXN Studios</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    time.sleep(2.5)  # Wait for animation to finish

# Clear the splash screen so the main app can load
splash.empty()

@st.dialog("⚙️ Settings & About")
def settings_modal():
    st.markdown("### 🚗 D.R.I.V.E.R.")
    st.caption("Document Retrieval & Intelligent Virtual Executive Researcher")
    st.divider()
    st.markdown(
        "**About this agent:**\n"
        "- **Orchestration:** LangChain `create_agent` (LangGraph runtime)\n"
        "- **Reasoning engine:** Gemini (swappable)\n"
        "- **Tools:** live web search, personal Google Drive search\n"
        "- **Memory:** persistent, SQLite-backed, survives restarts\n"
    )
    st.divider()
    st.markdown("**Developed by:** B.Rakshan")
    st.markdown("**Presented by:** RXN Studios")

# -----------------------------------------------------------------------
# Sidebar: Profile & Configuration
# -----------------------------------------------------------------------
with st.sidebar:
    st.header("Profile 👤")
    
    # 1. OAuth Scopes (Now includes profile and email data)
    AUTH_SCOPES = [
        "openid",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/drive.readonly"
    ]
    
    # In production, use your real domain (e.g., https://driver-app.streamlit.app)
    REDIRECT_URI = "https://driver-ragai.streamlit.app" 
    web_creds_path = os.environ.get("DRIVER_WEB_CREDENTIALS", "web_credentials.json")
    
    try:
        flow = Flow.from_client_secrets_file(
            web_creds_path,
            scopes=AUTH_SCOPES,
            redirect_uri=REDIRECT_URI
        )
    except Exception as e:
        flow = None
        st.error("⚠️ web_credentials.json not found. Check Google Cloud setup.")

    # 2. Catch the Redirect Code from Google
    if "code" in st.query_params and flow:
        try:
            flow.fetch_token(code=st.query_params["code"])
            st.session_state["user_creds"] = flow.credentials.to_json()
            st.query_params.clear()  # Clean the URL
            st.rerun()
        except Exception as e:
            st.error("Authentication failed.{e}")

    # 3. Render Profile or Login Button
    if "user_creds" in st.session_state:
        creds = Credentials.from_authorized_user_info(json.loads(st.session_state["user_creds"]))
        
        # Fetch the user's Google Profile data
        user_info = requests.get(
            "https://www.googleapis.com/oauth2/v1/userinfo", 
            headers={"Authorization": f"Bearer {creds.token}"}
        ).json()
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(user_info.get("picture", "https://via.placeholder.com/150"), width=50)
        with col2:
            st.markdown(f"**{user_info.get('name', 'User')}**")
            st.caption(user_info.get("email", ""))
            
        if st.button("Log Out", use_container_width=True):
            del st.session_state["user_creds"]
            st.rerun()
    else:
        if flow:
            auth_url, _ = flow.authorization_url(prompt='consent')
            st.link_button("🌐 Sign in with Google", auth_url, use_container_width=True)
          
    # Basic Settings
    st.subheader("")
    google_api_key = get_secret("GOOGLE_API_KEY")
    tavily_api_key = get_secret("TAVILY_API_KEY")
    if not google_api_key:
        google_api_key = st.text_input("GOOGLE_API_KEY", type="password")
    if not tavily_api_key:
        tavily_api_key = st.text_input("TAVILY_API_KEY", type="password")

    if st.button("⚙️ Settings & About", use_container_width=True):
        settings_modal()
    if st.button("🔄 New conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.last_msg_count = 0
        st.rerun()
      
    st.divider()
    st.subheader("💬 Chat History")
    saved_threads = get_saved_threads("driver_checkpoints.sqlite")
    if saved_threads:
        for t_id, t_title in saved_threads:
            # Highlight current active conversation
            button_label = f"▶ {t_title}" if t_id == st.session_state.get("thread_id") else t_title
            if st.button(button_label, key=f"hist_{t_id}", use_container_width=True):
                st.session_state.thread_id = t_id
                st.session_state.messages = load_messages_from_checkpoint(checkpointer, t_id)
                st.session_state.last_msg_count = len(st.session_state.messages)
                st.rerun()
    else:
        st.caption("No previous chats yet.")
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Guard: make sure we have the minimum required credentials before building
# the agent. The Drive tool degrades gracefully on its own (it returns an
# error string to the agent instead of crashing), so it isn't gated here.
# -----------------------------------------------------------------------
missing = []
if not google_api_key:
    missing.append("a Gemini `GOOGLE_API_KEY`")
if not tavily_api_key:
    missing.append("a `TAVILY_API_KEY`")

if missing:
    st.info(
        "Add " + " and ".join(missing) + " in the sidebar to start chatting."
    )
    st.stop()

# =============================================================================
# Session state: conversation memory
# =============================================================================
# The checkpointer itself is a process-wide singleton (see get_checkpointer
# above) so history survives restarts. Only the thread_id (which partitions
# one visitor's conversation from another's within that shared store) and
# the plain message list used to render the chat UI need to live per-session.

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_msg_count" not in st.session_state:
    st.session_state.last_msg_count = 0

checkpoint_db_path = "driver_checkpoints.sqlite"
checkpointer = get_checkpointer(checkpoint_db_path)

# =============================================================================
# Chat UI & Dynamic Agent Setup
# =============================================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Mode preference dropdown replaces the hardcoded model selector
col1, col2 = st.columns([1, 3])
with col1:
    preferred_mode = st.selectbox(
        "Mode Preference",
        options=["Auto-Detect (Recommended)", "Force: Standard (Flash)", "Force: High-Power (Pro)"],
        index=0,
        label_visibility="collapsed",
        help="AI will auto-route, but you can force a minimum model here."
    )

user_input = st.chat_input("Ask D.R.I.V.E.R. about your Drive or the web...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    thread_title = user_input[:28] + "..." if len(user_input) > 28 else user_input
    save_thread_title("driver_checkpoints.sqlite", st.session_state.thread_id, thread_title)
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        try:
            # 1. INTENT ROUTING
            with st.spinner("Analyzing intent..."):
                router_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", api_key=google_api_key)
                structured_router = router_llm.with_structured_output(TaskIntent)
                intent_result = structured_router.invoke(
                    f"Analyze this user request. Decide task_type (web, drive, hybrid, general) and complexity (low, high): {user_input}"
                )
                
                task_type = intent_result.task_type
                complexity = intent_result.complexity
            
            # 2. QUOTA & MODEL SELECTION
            quota_conn = setup_quota_db(checkpoint_db_path)
            pro_usage = get_pro_usage(quota_conn)
            
            active_model = "gemini-2.5-flash"
            dynamic_web_results = 3
            dynamic_drive_results = 5
            
            if preferred_mode == "Force: High-Power (Pro)" or (preferred_mode == "Auto-Detect (Recommended)" and (complexity == "high" or task_type == "hybrid")):
                if pro_usage < 5:
                    active_model = "gemini-1.5-pro"
                    dynamic_web_results = 8
                    dynamic_drive_results = 15
                    new_usage = increment_pro_usage(quota_conn)
                    st.caption(f"🚀 High-Power Mode Engaged (Daily Pro Uses: {new_usage}/5) | Intent: {task_type.upper()}")
                else:
                    st.warning("⚠️ Daily High-Power limit (5/5) reached. Automatically downgrading to Standard Mode.")
                    active_model = "gemini-2.5-flash"
                    dynamic_web_results = 5
                    dynamic_drive_results = 10
                    st.caption(f"🔍 Standard Mode | Intent: {task_type.upper()}")
            else:
                if task_type == "general":
                    active_model = "gemini-2.5-flash-lite"
                    st.caption(f"⚡ Fast Mode (No search needed)")
                else:
                    st.caption(f"🔍 Standard Mode | Intent: {task_type.upper()}")

            # 3. DYNAMIC TOOL BUILDING
            active_tools = []
            if task_type in ["web", "hybrid"]:
                active_tools.append(TavilySearch(max_results=dynamic_web_results, topic="general", tavily_api_key=tavily_api_key))
            
            if task_type in ["drive", "hybrid"]:
                if "user_creds" in st.session_state:
                    active_tools.append(build_drive_search_tool(st.session_state["user_creds"], dynamic_drive_results))
                else:
                    st.warning("⚠️ The AI attempted to search your Drive, but you are not logged in. Please sign in via the sidebar.")
            
            if not active_tools:
                active_tools.append(build_drive_search_tool(st.session_state.get("user_creds", "{}"), 1))
            
            exec_llm = ChatGoogleGenerativeAI(model=active_model, api_key=google_api_key)
            agent = create_react_agent(
                model=exec_llm,
                tools=active_tools,
                prompt=SYSTEM_PROMPT,
                checkpointer=checkpointer,
            )

            # 4. EXECUTION WITH TOKEN STREAMING
            st.session_state.last_msg_count = len(st.session_state.get("messages", []))
            
            tool_calls_made = []
            tool_outputs = []
            
            with st.status("D.R.I.V.E.R. is working...", expanded=False) as status:
                for chunk in agent.stream({"messages": [HumanMessage(content=user_input)]}, config=config, stream_mode="updates"):
                    for node_name, node_state in chunk.items():
                        if node_name == "agent":
                            latest_msg = node_state["messages"][-1]
                            if latest_msg.tool_calls:
                                for call in latest_msg.tool_calls:
                                    tool_calls_made.append(call)
                                    status.update(label=f"Calling tool: `{call['name']}`...", state="running")
                        elif node_name == "tools":
                            latest_msg = node_state["messages"][-1]
                            tool_outputs.append(latest_msg)
                            status.update(label="Processing tool results...", state="running")
                
                status.update(label="Synthesis complete", state="complete", expanded=False)

            final_state = agent.get_state(config)
            all_messages = final_state.values.get("messages", [])
            new_messages = all_messages[st.session_state.last_msg_count :]
            st.session_state.last_msg_count = len(all_messages)

            # 5. TRANSPARENCY PANEL
            if tool_calls_made:
                with st.expander("🔧 Tool activity", expanded=False):
                    for call in tool_calls_made:
                        st.markdown(f"**Called `{call['name']}`** — args: `{call['args']}`")
                    for m in tool_outputs:
                        preview = m.content
                        if len(preview) > 500:
                            preview = preview[:500] + "…"
                        st.markdown(f"**`{getattr(m, 'name', 'Tool')}` returned:**\n\n{preview}")

            # 6. RENDER STREAMED FINAL ANSWER
            final_answer = extract_text(all_messages[-1].content)
            
            def stream_response():
                for word in final_answer.split(" "):
                    yield word + " "
                    time.sleep(0.01)

            st.write_stream(stream_response())
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

        except Exception as exc:
            st.error(f"The agent hit an error: {exc}")
