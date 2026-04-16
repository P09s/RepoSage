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

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f1117;
    color: #e0e0e0;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #1e2535;
}
[data-testid="stHeader"] { background: transparent; }
footer { display: none !important; }

.block-container { padding: 2rem 2.5rem; }

.sources-bar {
    background: #0d1117;
    border: 1px solid #1e2535;
    border-radius: 6px;
    padding: 0.5rem 0.9rem;
    margin-top: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #4a6fa5;
}
.repo-badge {
    display: inline-block;
    background: #1a2236;
    border: 1px solid #2a3a5a;
    border-radius: 4px;
    padding: 0.25rem 0.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #3b82f6;
    margin-bottom: 1rem;
}
.page-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #f0f0f0;
    margin-bottom: 0.2rem;
}
.page-sub {
    font-size: 0.82rem;
    color: #4a5568;
    margin-bottom: 1.5rem;
}

/* Button */
.stButton > button {
    background: #1d4ed8 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.2rem !important;
    width: 100% !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #2563eb !important;
}
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
    """Extract 'owner/repo' from GitHub URL."""
    match = re.search(r"github\.com[/:]([^/]+)/([^/.]+)", url)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return url.rstrip("/").split("/")[-1]


def sanitize_collection_name(name: str) -> str:
    """Chroma collection names must be alphanumeric + hyphens."""
    return re.sub(r"[^a-zA-Z0-9-]", "-", name)[:60]


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### RepoSage")
    st.markdown("<div style='font-size:0.75rem;color:#4a5568;margin-bottom:1.5rem'>Chat with any GitHub codebase</div>", unsafe_allow_html=True)

    github_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repo",
    )
    github_token = st.text_input(
        "GitHub Token (for private repos)",
        type="password",
        placeholder="ghp_xxxxxxxxxxxx",
    )

    load_clicked = st.button("Load Repository")

    if load_clicked:
        if not github_url.strip():
            st.error("Enter a GitHub URL.")
        else:
            repo_name = extract_repo_name(github_url)
            collection = sanitize_collection_name(repo_name)

            with st.status("Loading repository...", expanded=True) as status:
                st.write("Cloning repository...")
                try:
                    repo_path = clone_repo(
                        github_url.strip(),
                        github_token.strip() if github_token.strip() else None,
                    )
                except ValueError as e:
                    st.error(str(e))
                    st.stop()

                st.write("Parsing and chunking codebase...")
                chunks = parse_repo_to_chunks(repo_path)

                if not chunks:
                    st.error("No supported files found in this repository.")
                    cleanup_repo(repo_path)
                    st.stop()

                st.write(f"Building vector index over {len(chunks)} chunks...")
                vectorstore = build_vectorstore(chunks, collection)
                chain, get_sources = build_rag_chain(vectorstore)

                cleanup_repo(repo_path)

                # Persist in session
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

                status.update(label="Repository loaded.", state="complete", expanded=False)

    # Stats
    if st.session_state.repo_name:
        st.markdown("---")
        st.markdown(f"<div class='repo-badge'>{st.session_state.repo_name}</div>", unsafe_allow_html=True)
        s = st.session_state.repo_stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", s.get("chunks", 0))
            st.metric("Languages", s.get("languages", 0))
        with col2:
            st.metric("Files", s.get("files", 0))

        if st.button("Clear & Load New Repo"):
            for key in ["chat_history", "vectorstore", "rag_chain", "get_sources", "repo_name", "repo_stats"]:
                st.session_state[key] = None if key not in ["chat_history", "repo_stats"] else ([] if key == "chat_history" else {})
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.68rem;color:#374151;line-height:1.7'>
    Stack: Llama 3.1 via Groq · HuggingFace · ChromaDB · LangChain · Streamlit
    </div>
    """, unsafe_allow_html=True)


# ── Main panel ────────────────────────────────────────────────
st.markdown("<div class='page-title'>RepoSage</div>", unsafe_allow_html=True)
st.markdown("<div class='page-sub'>Ask anything about the loaded repository</div>", unsafe_allow_html=True)

if not st.session_state.repo_name:
    st.info("Load a GitHub repository from the sidebar to start chatting.")
else:
    # Render chat history with native Streamlit components
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn['question'])
        with st.chat_message("assistant"):
            st.write(turn['answer'])
            if turn.get("sources"):
                sources_str = " · ".join(turn["sources"])
                st.markdown(f"<div class='sources-bar'>Sources: {sources_str}</div>", unsafe_allow_html=True)

    # Chat input and Suggestions logic
    question = st.chat_input("Ask about the codebase...")
    
    if not st.session_state.chat_history and not question:
        st.markdown("**Suggested questions to start:**")
        suggestions = [
            "Give me a high-level overview of this codebase",
            "What is the entry point of this application?",
            "Explain the data flow from input to output",
            "What are the main modules and what does each do?",
            "Are there any potential bugs or issues in the code?",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(s, key=f"suggest_{i}"):
                    question = s  # Bypass standard input and treat button click as question

    if question:
        # Show immediate user message
        with st.chat_message("user"):
            st.write(question)
            
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer..."):
                try:
                    answer = st.session_state.rag_chain.invoke(question)
                    sources = st.session_state.get_sources(question)

                    st.write(answer)
                    if sources:
                        sources_str = " · ".join(sources)
                        st.markdown(f"<div class='sources-bar'>Sources: {sources_str}</div>", unsafe_allow_html=True)
                        
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "sources": sources,
                    })
                except Exception as e:
                    st.error(f"Error: {str(e)}")