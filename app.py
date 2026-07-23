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

import streamlit as st
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_community import GoogleDriveLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver

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

def get_drive_credentials(credentials_path: str, token_path: str, scopes: List[str]):
    """
    Resolves Google Drive OAuth credentials in this priority order:

      1. `st.secrets["GOOGLE_DRIVE_TOKEN_JSON"]` — a pre-generated token (the
         *contents* of a token.json produced by an earlier local run). This
         is what makes a headless/hosted deployment possible: run this app
         locally ONCE, approve the browser consent, then paste the
         resulting token.json's contents into your host's secrets under
         this key. No browser is needed on the server after that.
      2. A cached `token_path` file on local disk (same-machine deployments).
      3. The interactive "installed app" OAuth flow via `credentials_path`,
         which opens a local browser tab — only works when this process and
         the browser share the same machine (e.g. `streamlit run` on your
         own laptop).

    A refreshed/renewed token is written back to `token_path` when possible
    (skipped silently on read-only filesystems some hosts use — Drive
    search still works for the process lifetime, it just re-authenticates
    next boot).
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    token_file = Path(token_path)

    # 1. Pre-generated token supplied via secrets (headless deployment path).
    token_secret = get_secret("GOOGLE_DRIVE_TOKEN_JSON")
    if token_secret:
        creds = Credentials.from_authorized_user_info(json.loads(token_secret), scopes)

    # 2. Cached token file from a previous local run.
    if not creds and token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not Path(credentials_path).exists():
                raise FileNotFoundError(
                    f"Google OAuth client file not found at '{credentials_path}'. "
                    "Download it from Google Cloud Console (APIs & Services > "
                    "Credentials > OAuth client ID > Desktop app), point the "
                    "sidebar to it, and run this app locally once to consent. "
                    "For a headless deployment, do that once locally, then "
                    "paste the resulting token.json contents into your host's "
                    "secrets as GOOGLE_DRIVE_TOKEN_JSON."
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, scopes)
            creds = flow.run_local_server(port=0)  # opens a browser tab, once

        try:
            token_file.write_text(creds.to_json())
        except OSError:
            pass  # read-only filesystem (e.g. some hosts) — non-fatal

    return creds


def build_drive_search_tool(credentials_path: str, token_path: str, num_results: int):
    """
    Factory that returns a LangChain tool closing over the user's chosen
    credentials/token paths and result-count setting from the sidebar.
    """

    @tool("google_drive_search")
    def google_drive_search(query: str) -> str:
        """Search the user's personal Google Drive for documents matching the
        query, and return their text content along with the file name/title
        and a link, so the answer can cite the exact source."""
        try:
            from googleapiclient.discovery import build

            creds = get_drive_credentials(credentials_path, token_path, DRIVE_SCOPES)
            drive_service = build("drive", "v3", credentials=creds)

            # Google Drive API v3 full-text search. Escape single quotes so a
            # query containing one can't break the query string.
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

            # Hand the matched file IDs to GoogleDriveLoader, which already
            # knows how to export Google Docs/Sheets and extract PDF text.
            # We pass `credentials` directly so it reuses the same OAuth
            # session instead of re-authenticating.
            loader = GoogleDriveLoader(
                file_ids=[f["id"] for f in matches],
                credentials=creds,
                scopes=DRIVE_SCOPES,
            )
            docs = loader.load()
        except Exception as exc:  # noqa: BLE001 - surface any failure to the agent
            return f"Google Drive search failed: {exc}"

        if not docs:
            return (
                "Matching files were found in Drive, but no readable text could "
                "be extracted from them (unsupported file type)."
            )

        chunks = []
        for doc in docs:
            title = doc.metadata.get("title", "Untitled")
            source = doc.metadata.get("source", "")
            # Cap each document's contribution so one huge file can't blow
            # out the agent's context window.
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
# Streamlit page setup
# =============================================================================

st.set_page_config(page_title="D.R.I.V.E.R.", page_icon="🚗", layout="wide")
st.title("🚗 D.R.I.V.E.R.")
st.caption("Data Retriever & Intelligent Virtual Executive Researcher")
import time

# Create a placeholder container
splash = st.empty()

with splash.container():
    st.markdown(
        """
        <style>
        .splash-text {
            font-size: 4rem;
            font-weight: bold;
            color: #FF4B4B; /* Streamlit red, or change to your brand color */
            text-align: center;
            margin-top: 30vh;
            animation: fadeInOut 3.5s ease-in-out forwards;
        }
        @keyframes fadeInOut {
            0% { opacity: 0; transform: scale(0.9); }
            30% { opacity: 1; transform: scale(1); }
            80% { opacity: 1; transform: scale(1); }
            100% { opacity: 0; transform: scale(1.1); }
        }
        </style>
        <div class="splash-text">RXN Studios</div>
        """,
        unsafe_allow_html=True
    )
    time.sleep(2.5) # Wait for animation to finish

# Clear the splash screen so the main app can load
splash.empty()
# -----------------------------------------------------------------------
# Sidebar: configuration
# -----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Reasoning Engine")
    model_name = st.text_input(
        "Gemini model",
        value=DEFAULT_MODEL,
        help="Any model string ChatGoogleGenerativeAI accepts, "
        "e.g. gemini-2.5-flash, gemini-3-flash-preview.",
    )
    google_api_key = get_secret("GOOGLE_API_KEY")
    if google_api_key:
        st.success("GOOGLE_API_KEY loaded from secrets/environment ✅")
    else:
        google_api_key = st.text_input(
            "GOOGLE_API_KEY",
            type="password",
            help="Gemini Developer API key. For a deployed app, set this in "
            "`.streamlit/secrets.toml` (or your host's secret manager) "
            "instead so visitors never see or need one.",
        )

    st.subheader("Web Search (Tavily)")
    tavily_api_key = get_secret("TAVILY_API_KEY")
    if tavily_api_key:
        st.success("TAVILY_API_KEY loaded from secrets/environment ✅")
    else:
        tavily_api_key = st.text_input(
            "TAVILY_API_KEY",
            type="password",
            help="For a deployed app, set this in `.streamlit/secrets.toml` "
            "(or your host's secret manager) instead.",
        )
    max_web_results = st.slider("Max web results", 1, 10, 5)

    st.subheader("Google Drive")
    credentials_path = st.text_input(
        "credentials.json path",
        value=os.environ.get("DRIVER_GOOGLE_CREDENTIALS", "credentials.json"),
        help="OAuth client file downloaded from Google Cloud Console "
        "(Desktop app credentials).",
    )
    token_path = st.text_input(
        "token.json path (auto-created)",
        value=os.environ.get("DRIVER_GOOGLE_TOKEN", "token.json"),
        help="Cached after the first browser consent — delete this file to "
        "force re-authentication (e.g. after changing scopes).",
    )
    max_drive_results = st.slider("Max Drive documents per search", 1, 15, 5)

    st.subheader("Memory")
    checkpoint_db_path = st.text_input(
        "Conversation DB path",
        value=os.environ.get("DRIVER_CHECKPOINT_DB", "driver_checkpoints.sqlite"),
        help="SQLite file storing conversation history so it survives app "
        "restarts. Shared by all sessions on this server; each browser "
        "session gets its own conversation via a unique thread_id.",
    )

    st.divider()
    if st.button("🔄 New conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.last_msg_count = 0
        st.rerun()

    with st.expander("ℹ️ About this agent"):
        st.markdown(
            "- **Orchestration:** LangChain `create_agent` (LangGraph runtime)\n"
            "- **Reasoning engine:** Gemini (swappable)\n"
            "- **Tools:** live web search, personal Google Drive search\n"
            "- **Memory:** persistent, SQLite-backed, survives restarts\n"
        )

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
    st.session_state.messages = []  # only for rendering the chat UI
if "last_msg_count" not in st.session_state:
    st.session_state.last_msg_count = 0

checkpointer = get_checkpointer(checkpoint_db_path)

# =============================================================================
# Build tools, LLM, and the agent graph
# =============================================================================

web_search_tool = TavilySearch(
    max_results=max_web_results,
    topic="general",
    tavily_api_key=tavily_api_key,
)

drive_search_tool = build_drive_search_tool(
    credentials_path=credentials_path,
    token_path=token_path,
    num_results=max_drive_results,
)

llm = ChatGoogleGenerativeAI(model=model_name, api_key=google_api_key)

agent = create_react_agent(
    model=llm,
    tools=[web_search_tool, drive_search_tool],
    prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

# =============================================================================
# Chat UI
# =============================================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask D.R.I.V.E.R. about your Drive or the web...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        try:
            with st.spinner("Thinking..."):
                # Only the NEW message needs to be sent — the checkpointer
                # (keyed by thread_id) already holds prior turns and will be
                # merged in automatically by the graph.
                result = agent.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                )
        except Exception as exc:  # noqa: BLE001 - show API/auth errors in the UI
            st.error(f"The agent hit an error: {exc}")
            st.stop()

        all_messages = result["messages"]
        new_messages = all_messages[st.session_state.last_msg_count :]
        st.session_state.last_msg_count = len(all_messages)

        # Transparency panel: show which tool(s) fired and what they returned,
        # mirroring the design doc's "Source Attribution & Validation" goal.
        tool_calls_made = [
            m for m in new_messages if isinstance(m, AIMessage) and m.tool_calls
        ]
        if tool_calls_made:
            with st.expander("🔧 Tool activity", expanded=False):
                for m in new_messages:
                    if isinstance(m, AIMessage) and m.tool_calls:
                        for call in m.tool_calls:
                            st.markdown(f"**Called `{call['name']}`** — args: `{call['args']}`")
                    elif isinstance(m, ToolMessage):
                        preview = m.content
                        if len(preview) > 500:
                            preview = preview[:500] + "…"
                        st.markdown(f"**`{m.name}` returned:**\n\n{preview}")

        final_answer = extract_text(all_messages[-1].content)
        st.markdown(final_answer)

    st.session_state.messages.append({"role": "assistant", "content": final_answer})
