import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

CHROMA_DIR = "./chroma_store"


# ── Singletons ────────────────────────────────────────────────
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.2,
        max_tokens=2048,
    )


# ── Vector store ───────────────────────────────────────────────
def build_vectorstore(chunks: list[dict], repo_name: str) -> Chroma:
    """
    Convert raw chunk dicts to LangChain Documents and persist in Chroma.
    Uses repo_name as the collection name so multiple repos can coexist.
    """
    documents = []
    for chunk in chunks:
        # Format content so the LLM gets full context about where this came from
        content = (
            f"File: {chunk['filepath']}\n"
            f"Language: {chunk['language']}\n"
            f"Type: {chunk.get('chunk_type', 'block')}\n"
            f"{'Name: ' + chunk['name'] + chr(10) if 'name' in chunk else ''}"
            f"\n{chunk['content']}"
        )
        doc = Document(
            page_content=content,
            metadata={
                "filepath": chunk["filepath"],
                "language": chunk["language"],
                "chunk_type": chunk.get("chunk_type", "block"),
                "name": chunk.get("name", ""),
                "start_line": chunk.get("start_line", 0),
            }
        )
        documents.append(doc)

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=repo_name,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore


def load_vectorstore(repo_name: str) -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        collection_name=repo_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


# ── Multi-hop retrieval ────────────────────────────────────────
def multihop_retrieve(vectorstore: Chroma, question: str, k: int = 6) -> list[Document]:
    """
    Step 1: Retrieve top-k chunks for the question.
    Step 2: Extract any filenames mentioned in those chunks.
    Step 3: Retrieve additional chunks from those related files.
    This simulates reading connected files, not just keyword matches.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    initial_docs = retriever.invoke(question)

    # Collect related filepaths from initial results
    related_files = set()
    for doc in initial_docs:
        fp = doc.metadata.get("filepath", "")
        if fp:
            related_files.add(fp)

    # Second hop: fetch more chunks from those same files
    second_hop_docs = []
    if related_files:
        all_docs = vectorstore.get()
        if all_docs and "documents" in all_docs:
            for i, content in enumerate(all_docs["documents"]):
                meta = all_docs["metadatas"][i]
                if meta.get("filepath") in related_files:
                    second_hop_docs.append(
                        Document(page_content=content, metadata=meta)
                    )

    # Merge, deduplicate by content
    seen = set()
    merged = []
    for doc in initial_docs + second_hop_docs:
        key = doc.page_content[:120]
        if key not in seen:
            seen.add(key)
            merged.append(doc)

    return merged[:10]  # cap at 10 chunks for context window


# ── RAG chain ──────────────────────────────────────────────────
SYSTEM_PROMPT = """You are RepoSage, an expert AI assistant that deeply understands codebases.

You are given retrieved code chunks from a GitHub repository. Answer the user's question
accurately and concisely. When referencing code, mention the file path and function/class name.
If the answer spans multiple files, explain the flow across them.
If you cannot find the answer in the provided context, say so clearly — do not hallucinate.

Context from codebase:
{context}
"""

def build_rag_chain(vectorstore: Chroma):
    """Return a callable LCEL chain: question -> answer string."""
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    def retrieve_and_format(question: str) -> str:
        docs = multihop_retrieve(vectorstore, question)
        return "\n\n---\n\n".join([doc.page_content for doc in docs])

    def get_sources(question: str) -> list[str]:
        docs = multihop_retrieve(vectorstore, question)
        return list({doc.metadata.get("filepath", "") for doc in docs})

    chain = (
        {
            "context": retrieve_and_format,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, get_sources