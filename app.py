import streamlit as st
import os
import re
from dotenv import load_dotenv
from repo_ingestion import clone_repo, parse_repo_to_chunks, cleanup_repo
from rag_engine import build_vectorstore, load_vectorstore, build_rag_chain, CHROMA_DIR

load_dotenv()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="RepoSage",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SVG Icon Library (Lucide) ─────────────────────────────────
ICONS = {
    "brain": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.46 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.14"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.46 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24A2.5 2.5 0 0 0 14.5 2"/></svg>""",
    "github": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.2c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/></svg>""",
    "key": """<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="m15.5 7.5 2.3 2.3a1 1 0 0 0 1.4 0l2.1-2.1a1 1 0 0 0 0-1.4L19 4"/><path d="m21 2-9.6 9.6"/><circle cx="7.5" cy="15.5" r="5.5"/></svg>""",
    "layers": """<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 12.5-8.55 3.9a2 2 0 0 1-1.66 0l-8.58-3.9"/><path d="m22 17.5-8.55 3.9a2 2 0 0 1-1.66 0l-8.58-3.9"/></svg>""",
    "file_code": """<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M10 12.5 8 15l2 2.5"/><path d="m14 12.5 2 2.5-2 2.5"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7z"/></svg>""",
    "cpu": """<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/></svg>""",
    "terminal": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 17 10 11 4 5"/><line x1="12" x2="20" y1="19" y2="19"/></svg>""",
    "git_branch": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><line x1="6" x2="6" y1="3" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>""",
    "alert_circle": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/></svg>""",
}

# ── CSS ───────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&family=IBM+Plex+Mono:ital,wght@0,400;0,500;1,400&display=swap');

html, body, [data-testid="stAppViewContainer"] {{
    background-color: #080c12 !important;
    color: #c9d1d9;
    font-family: 'Syne', sans-serif;
}}

/* ─ FIX: Sidebar Layout & Top Padding ─ */
[data-testid="stSidebar"] {{
    background-color: #0c1018 !important;
    border-right: 1px solid #1c2333 !important;
}}

/* Remove default whitespace at the very top of the sidebar */
[data-testid="stSidebarHeader"] {{
    padding-top: 1rem !important;
    padding-bottom: 0rem !important;
}}

[data-testid="stSidebarUserContent"] {{
    padding-top: 0rem !important;
}}

/* ─ FIX: Sidebar Collapse/Expand Toggle Icon Visibility ─ */
/* Makes the ">" icon clearly visible when sidebar is closed */
[data-testid="collapsedControl"] {{
    background-color: #0c1018 !important;
    border: 1px solid #1c2333 !important;
    border-radius: 8px !important;
    color: #cbd5e1 !important;
    top: 1rem !important;
    left: 1rem !important;
    z-index: 1000 !important;
    padding: 0.2rem !important;
}}

[data-testid="collapsedControl"]:hover {{
    background-color: #111827 !important;
    color: #fff !important;
}}

/* Ensure the "X" inside the open sidebar is positioned correctly so it doesn't overlap logo */
[data-testid="stSidebarHeader"] button {{
    position: absolute;
    top: 15px;
    right: 15px;
    color: #94a3b8 !important;
}}

[data-testid="stHeader"] {{ background: transparent !important; }}
footer {{ display: none !important; }}
#MainMenu {{ display: none !important; }}
.block-container {{ padding: 1.5rem 2rem 2rem !important; max-width: 900px; }}

/* ─ Sidebar Inputs ─ */
[data-testid="stSidebar"] .stTextInput input {{
    background: #0f1623 !important;
    border: 1px solid #1c2333 !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    padding: 10px 14px !important;
    transition: all 0.2s ease !important;
}}

[data-testid="stSidebar"] .stTextInput input:focus {{
    border-color: #3b82f6 !important;
    color: #e2e8f0 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
    background: #111827 !important;
}}

/* ─ Sidebar Buttons ─ */
[data-testid="stSidebar"] button[kind="primary"] {{
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 12px 0 !important;
    width: 100% !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
}}

[data-testid="stSidebar"] button[kind="primary"]:hover {{
    background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(29,78,216,0.3) !important;
}}

[data-testid="stSidebar"] button[kind="secondary"] {{
    background: #0f1623 !important;
    color: #f87171 !important;
    border: 1px solid #7f1d1d !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 500 !important;
    font-size: 12px !important;
    padding: 10px 0 !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}}

[data-testid="stSidebar"] button[kind="secondary"]:hover {{
    background: #450a0a !important;
    border-color: #ef4444 !important;
    color: #fca5a5 !important;
}}

/* ─ Main Area Buttons (Suggestions) ─ */
div[data-testid="stMainBlockContainer"] .stButton > button {{
    background: #0c1018 !important;
    border: 1px solid #1c2333 !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 16px 20px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 75px !important;
    white-space: normal !important;
    line-height: 1.5 !important;
    display: flex !important;
    align-items: center !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
}}

div[data-testid="stMainBlockContainer"] .stButton > button p {{
    margin: 0 !important;
    font-size: 13px !important;
}}

div[data-testid="stMainBlockContainer"] .stButton > button:hover {{
    border-color: #3b82f6 !important;
    background: #111827 !important;
    color: #e2e8f0 !important;
    box-shadow: 0 6px 12px rgba(59, 130, 246, 0.1) !important;
    transform: translateY(-2px) !important;
}}

/* ─ Metric cards ─ */
[data-testid="stMetric"] {{
    background: #0f1623 !important;
    border: 1px solid #1c2333 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    text-align: center !important;
}}

[data-testid="stMetricLabel"] {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 0.1em !important;
    color: #64748b !important;
    text-transform: uppercase !important;
    justify-content: center !important;
}}

[data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #f1f5f9 !important;
}}

/* ─ Chat input ─ */
[data-testid="stChatInput"] textarea {{
    background: #0c1018 !important;
    border: 1px solid #1c2333 !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 14px !important;
    padding: 14px !important;
}}

[data-testid="stChatInput"] textarea:focus {{
    border-color: #1d4ed888 !important;
    box-shadow: 0 0 0 3px rgba(29,78,216,0.15) !important;
}}

/* ─ Chat messages ─ */
[data-testid="stChatMessage"] {{
    background: transparent !important;
    border: none !important;
    padding: 0 0 20px 0 !important;
}}

[data-testid="stChatMessageContent"] {{
    font-family: 'Syne', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    color: #cbd5e1 !important;
}}

/* ─ Custom components ─ */
.rs-logo-block {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0px 20px 16px; /* Removed top padding to pull it upwards */
    border-bottom: 1px solid #1c2333;
    margin-bottom: 16px;
    margin-top: -12px; /* Pull it slightly into the newly created header gap */
}}

.rs-logo-icon {{
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    color: #bfdbfe;
    flex-shrink: 0;
}}

.rs-logo-text {{
    font-size: 18px;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: 0.02em;
    line-height: 1.1;
}}

.rs-logo-sub {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: #64748b;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 2px;
}}

.rs-section-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.12em;
    color: #64748b;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 6px;
}}

.rs-repo-card {{
    background: #0f1623;
    border: 1px solid #1c2333;
    border-radius: 9px;
    padding: 14px;
    margin-bottom: 10px;
}}

.rs-repo-header {{
    display: flex;
    align-items: center;
    gap: 8px;
}}

.rs-status-live {{
    width: 8px; height: 8px;
    background: #22c55e;
    border-radius: 50%;
    box-shadow: 0 0 8px #22c55e88;
    animation: rs-pulse 2s ease-in-out infinite;
    flex-shrink: 0;
}}

@keyframes rs-pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.6; transform: scale(0.85); }}
}}

.rs-repo-name {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #60a5fa;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}

.rs-page-title {{
    font-size: 24px;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: 0.01em;
    margin-bottom: 4px;
}}

.rs-page-sub {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #64748b;
    letter-spacing: 0.05em;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.rs-status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(20, 83, 45, 0.4);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 20px;
    padding: 4px 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #4ade80;
    letter-spacing: 0.05em;
}}

.rs-suggestions-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.1em;
    color: #475569;
    text-transform: uppercase;
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}}

.rs-suggestions-label::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: #1e293b;
}}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "get_sources" not in st.session_state:
    st.session_state.get_sources = None
if "repo_name" not in st.session_state:
    st.session_state.repo_name = None
if "repo_stats" not in st.session_state:
    st.session_state.repo_stats = {}

def extract_repo_name(url: str) -> str:
    match = re.search(r"github\.com[/:]([^/]+)/([^/.]+)", url)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return url.rstrip("/").split("/")[-1]

def sanitize_collection_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9-]", "-", name)[:60]

def make_sources_html(sources: list) -> str:
    tags = "".join(
        f'<span style="background:#0f172a; border:1px solid #1e293b; border-radius:4px; padding:3px 8px; font-family:\'IBM Plex Mono\', monospace; font-size:10px; color:#93c5fd; display:inline-flex; align-items:center; gap:5px; margin-right:6px; margin-bottom:6px;">{ICONS["file_code"]} {s}</span>'
        for s in sources
    )
    return f"""
    <div style="margin-top:12px; padding-top:12px; border-top:1px solid #1e293b;">
      <div style="font-family:'IBM Plex Mono', monospace; font-size:9px; letter-spacing:0.1em; color:#475569; text-transform:uppercase; margin-bottom:8px; display:flex; align-items:center; gap:5px;">
        {ICONS["layers"]} Sources
      </div>
      <div>{tags}</div>
    </div>
    """


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:

    # Logo block
    st.markdown(f"""
    <div class="rs-logo-block">
      <div class="rs-logo-icon">
        {ICONS["brain"].replace('width="16" height="16"', 'width="20" height="20"')}
      </div>
      <div>
        <div class="rs-logo-text">RepoSage</div>
        <div class="rs-logo-sub">Codebase Intelligence</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown(f"""
        <div class="rs-section-label" style="padding: 0 5px;">
          {ICONS["github"]} Repository URL
        </div>
        """, unsafe_allow_html=True)
        github_url = st.text_input(
            "GitHub Repository URL",
            placeholder="https://github.com/owner/repo",
            label_visibility="collapsed",
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="rs-section-label" style="padding: 0 5px;">
          {ICONS["key"]} Access Token (private repo)
        </div>
        """, unsafe_allow_html=True)
        github_token = st.text_input(
            "GitHub Token",
            type="password",
            placeholder="ghp_xxxxxxxxxxxx",
            label_visibility="collapsed",
        )
        
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        
        load_clicked = st.button("Load Repository", type="primary", use_container_width=True)

    # ─ Load logic ─
    if load_clicked:
        if not github_url.strip():
            st.error("Enter a GitHub repository URL.")
        else:
            repo_name = extract_repo_name(github_url)
            collection = sanitize_collection_name(repo_name)

            with st.status("Indexing repository...", expanded=True) as status:
                st.write("Cloning repository...")
                try:
                    repo_path = clone_repo(
                        github_url.strip(),
                        github_token.strip() if github_token.strip() else None,
                    )
                except ValueError as e:
                    st.error(str(e))
                    st.stop()

                st.write("Parsing and chunking source files...")
                chunks = parse_repo_to_chunks(repo_path)

                if not chunks:
                    st.error("No supported files found in this repository.")
                    cleanup_repo(repo_path)
                    st.stop()

                st.write(f"Building vector index over {len(chunks):,} chunks...")
                vectorstore = build_vectorstore(chunks, collection)
                chain, get_sources = build_rag_chain(vectorstore)

                cleanup_repo(repo_path)

                st.session_state.vectorstore = vectorstore
                st.session_state.rag_chain = chain
                st.session_state.get_sources = get_sources
                st.session_state.repo_name = repo_name
                st.session_state.chat_history = []
                st.session_state.repo_stats = {
                    "chunks": len(chunks),
                    "files": len(set(c["filepath"] for c in chunks)),
                    "languages": len(set(c["language"] for c in chunks)),
                }

                status.update(label="Repository indexed.", state="complete", expanded=False)

    # ─ Repo stats card ─
    if st.session_state.repo_name:
        st.markdown("<hr style='border-color:#1e293b; margin: 24px 0;'/>", unsafe_allow_html=True)
        s = st.session_state.repo_stats
        
        st.markdown(f"""
        <div class="rs-repo-card">
          <div class="rs-repo-header">
            <div class="rs-status-live"></div>
            <div class="rs-repo-name" title="{st.session_state.repo_name}">{st.session_state.repo_name}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Chunks", f"{s.get('chunks', 0):,}")
        with col2:
            st.metric("Files", f"{s.get('files', 0):,}")
        with col3:
            st.metric("Langs", f"{s.get('languages', 0)}")

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        
        if st.button("Clear Session", type="secondary", use_container_width=True):
            for key in ["chat_history", "vectorstore", "rag_chain", "get_sources", "repo_name", "repo_stats"]:
                st.session_state[key] = (
                    [] if key == "chat_history" else
                    {} if key == "repo_stats" else
                    None
                )
            st.rerun()

    # ─ Stack ─
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="padding: 0 5px;">
      <div class="rs-section-label">{ICONS["cpu"]} Tech Stack</div>
      <div style="display:flex; flex-wrap:wrap; gap:6px; margin-top:10px;">
        <span style="background:#0f172a; border:1px solid #1e293b; border-radius:4px; padding:4px 8px; font-family:'IBM Plex Mono', monospace; font-size:9px; color:#64748b;">Llama 3.1</span>
        <span style="background:#0f172a; border:1px solid #1e293b; border-radius:4px; padding:4px 8px; font-family:'IBM Plex Mono', monospace; font-size:9px; color:#64748b;">Groq</span>
        <span style="background:#0f172a; border:1px solid #1e293b; border-radius:4px; padding:4px 8px; font-family:'IBM Plex Mono', monospace; font-size:9px; color:#64748b;">ChromaDB</span>
        <span style="background:#0f172a; border:1px solid #1e293b; border-radius:4px; padding:4px 8px; font-family:'IBM Plex Mono', monospace; font-size:9px; color:#64748b;">LangChain</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main panel ────────────────────────────────────────────────
if st.session_state.repo_name:
    status_html = f"""
    <span class="rs-status-pill">
      <span style="width:6px;height:6px;background:#4ade80;border-radius:50%;display:inline-block;box-shadow:0 0 6px #4ade80;"></span>
      INDEXED
    </span>
    """
else:
    status_html = ""

st.markdown(f"""
<div class="rs-page-title">RepoSage</div>
<div class="rs-page-sub">
  {ICONS["terminal"]} &nbsp;Ask anything about your codebase &nbsp;{status_html}
</div>
""", unsafe_allow_html=True)

# ─ Empty state ─
if not st.session_state.repo_name:
    st.markdown(f"""
    <div class="rs-empty-state">
      <div class="rs-empty-icon">
        {ICONS["github"].replace('width="14" height="14"', 'width="42" height="42"')}
      </div>
      <div style="font-size: 16px; font-weight: 600; color: #94a3b8; margin-bottom: 8px;">No repository loaded</div>
      <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #475569;">Paste a GitHub URL in the sidebar to begin indexing your codebase.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    # ─ Chat history ─
    for turn in st.session_state.chat_history:
        with st.chat_message("user", avatar=None):
            st.write(turn["question"])
        with st.chat_message("assistant", avatar=None):
            st.write(turn["answer"])
            if turn.get("sources"):
                st.markdown(make_sources_html(turn["sources"]), unsafe_allow_html=True)

    # ─ Chat Input Component ─
    question = st.chat_input("Ask about the codebase…")

    # ─ Suggestions (first visit) ─
    if not st.session_state.chat_history and not question:
        st.markdown(f"""
        <div class="rs-suggestions-label">
          {ICONS["git_branch"]} Suggested queries
        </div>
        """, unsafe_allow_html=True)

        suggestions = [
            ("Give me a high-level overview of this codebase"),
            ("What is the entry point of this application?"),
            ("Explain the data flow from input to output"),
            ("What are the main modules and what does each do?"),
            ("Are there any potential bugs or security issues?"),
            ("What external dependencies does this project use?"),
        ]

        cols = st.columns(2, gap="medium")
        for i, full_q in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(full_q, key=f"suggest_{i}", use_container_width=True):
                    question = full_q

    # ─ Process question ─
    if question:
        with st.chat_message("user", avatar=None):
            st.write(question)

        with st.chat_message("assistant", avatar=None):
            with st.spinner("Analyzing codebase..."):
                try:
                    answer = st.session_state.rag_chain.invoke(question)
                    sources = st.session_state.get_sources(question)

                    st.write(answer)
                    if sources:
                        st.markdown(make_sources_html(sources), unsafe_allow_html=True)

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "sources": sources,
                    })
                except Exception as e:
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:10px;color:#f87171;
                                font-family:'IBM Plex Mono',monospace;font-size:12px;
                                padding:12px 16px;background:#450a0a;border:1px solid #7f1d1d;
                                border-radius:8px; margin-top:12px;">
                      {ICONS["alert_circle"]} {str(e)}
                    </div>
                    """, unsafe_allow_html=True)