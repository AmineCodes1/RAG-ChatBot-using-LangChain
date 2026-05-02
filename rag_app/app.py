"""Streamlit application for document ingestion and RAG-based Q&A."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Any, TypedDict

import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from loguru import logger

from config import CHROMA_PERSIST_DIR, TOP_K
from ingestion.embedder import run_ingestion_pipeline
from retrieval.chain import build_rag_chain, query_rag
from retrieval.retriever import get_retriever, get_vectorstore

st.set_page_config(page_title="RAG Agent", page_icon="📄", layout="wide")


class ChatMessage(TypedDict):
    """Typed chat message model stored in session state."""

    role: str
    content: str
    sources: list[str]


def _initialize_session_state() -> None:
    """Initialize all required Streamlit session state keys."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "chain" not in st.session_state:
        st.session_state["chain"] = None
    if "ingestion_done" not in st.session_state:
        st.session_state["ingestion_done"] = False
    if "chain_collection_hash" not in st.session_state:
        st.session_state["chain_collection_hash"] = None
    if "last_upload_signature" not in st.session_state:
        st.session_state["last_upload_signature"] = None


@st.cache_resource
def _get_cached_vectorstore() -> Chroma:
    """Return a cached Chroma vectorstore instance."""
    return get_vectorstore()


def _get_collection_count(vectorstore: Chroma) -> int:
    """Get the number of indexed records from the underlying Chroma collection.

    Args:
        vectorstore: Initialized Chroma vectorstore.

    Returns:
        Number of indexed documents/chunks in the collection.
    """
    collection = getattr(vectorstore, "_collection", None)
    if collection is None:
        return 0
    return int(collection.count())


def _save_uploaded_files(uploaded_files: list[Any], target_dir: Path) -> list[str]:
    """Persist uploaded files to a target directory.

    Args:
        uploaded_files: Streamlit uploaded file objects.
        target_dir: Destination directory for saved files.

    Returns:
        Paths of saved files as strings.
    """
    file_paths: list[str] = []
    for uploaded_file in uploaded_files:
        destination_path = target_dir / uploaded_file.name
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(uploaded_file.getvalue())
        file_paths.append(str(destination_path))
    return file_paths


def _upload_signature(uploaded_files: list[Any]) -> str:
    """Build a deterministic signature for an uploaded file batch."""
    signature_payload = "|".join(
        sorted(f"{uploaded_file.name}:{uploaded_file.size}" for uploaded_file in uploaded_files)
    )
    return hashlib.md5(signature_payload.encode("utf-8")).hexdigest()


def _reset_for_new_index() -> None:
    """Reset state keys that depend on index contents."""
    st.session_state["chain"] = None
    st.session_state["chain_collection_hash"] = None
    st.session_state["ingestion_done"] = True


def _render_sources(sources: list[str]) -> None:
    """Render answer sources in an expander."""
    with st.expander("Sources"):
        for source in sources:
            st.markdown(f"- {source}")


def _build_or_get_chain(vectorstore: Chroma, collection_count: int) -> RetrievalQA:
    """Build a new chain only when the index hash changed.

    Args:
        vectorstore: Chroma vectorstore.
        collection_count: Current indexed collection count.

    Returns:
        A RetrievalQA chain instance.
    """
    current_hash = hashlib.md5(str(collection_count).encode("utf-8")).hexdigest()
    cached_chain = st.session_state.get("chain")
    cached_hash = st.session_state.get("chain_collection_hash")

    if cached_chain is None or cached_hash != current_hash:
        logger.info(
            "Rebuilding RAG chain (cached_hash='{}', current_hash='{}').",
            cached_hash,
            current_hash,
        )
        retriever = get_retriever(vectorstore, top_k=TOP_K)
        st.session_state["chain"] = build_rag_chain(retriever)
        st.session_state["chain_collection_hash"] = current_hash

    return st.session_state["chain"]


def _render_chat_history(messages: list[ChatMessage]) -> None:
    """Render chat history from session state."""
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                _render_sources(message["sources"])


def _handle_sidebar(collection_count: int) -> None:
    """Render and process all sidebar actions.

    Args:
        collection_count: Current collection count for stats display.
    """
    st.sidebar.title("Document Management")
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        batch_signature = _upload_signature(uploaded_files)
        if st.session_state["last_upload_signature"] != batch_signature:
            temp_dir = Path(tempfile.mkdtemp(prefix="rag_upload_"))
            try:
                with st.spinner("Ingesting uploaded documents..."):
                    file_paths = _save_uploaded_files(uploaded_files, temp_dir)
                    stats = run_ingestion_pipeline(file_paths)
                _get_cached_vectorstore.clear()
                _reset_for_new_index()
                st.session_state["last_upload_signature"] = batch_signature
                st.success(
                    (
                        "Ingestion complete — "
                        f"files_processed={stats['files_processed']}, "
                        f"chunks_added={stats['chunks_added']}, "
                        f"chunks_skipped={stats['chunks_skipped']}"
                    )
                )
            except Exception as exc:
                logger.exception("File ingestion failed.")
                st.error(f"Ingestion error: {exc}")
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        st.session_state["last_upload_signature"] = None

    if st.sidebar.button("Clear vector index"):
        try:
            shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)
            _get_cached_vectorstore.clear()
            st.session_state["messages"] = []
            st.session_state["chain"] = None
            st.session_state["ingestion_done"] = False
            st.session_state["chain_collection_hash"] = None
            st.session_state["last_upload_signature"] = None
            st.warning("Vector index cleared.")
            st.rerun()
        except Exception as exc:
            logger.exception("Failed clearing vector index.")
            st.error(f"Failed to clear vector index: {exc}")

    if st.sidebar.button("Clear chat"):
        st.session_state["messages"] = []
        st.sidebar.success("Chat cleared.")

    st.sidebar.markdown(f"**Indexed chunks:** {collection_count}")


def main() -> None:
    """Run the complete Streamlit RAG application."""
    _initialize_session_state()

    vectorstore = _get_cached_vectorstore()
    try:
        collection_count = _get_collection_count(vectorstore)
    except Exception as exc:
        logger.exception("Unable to read collection statistics.")
        collection_count = 0
        st.error(f"Failed to read index stats: {exc}")

    _handle_sidebar(collection_count)

    vectorstore = _get_cached_vectorstore()
    try:
        collection_count = _get_collection_count(vectorstore)
    except Exception as exc:
        logger.exception("Unable to refresh collection statistics.")
        collection_count = 0
        st.error(f"Failed to refresh index stats: {exc}")

    st.header("RAG Agent — Ask your documents")
    _render_chat_history(st.session_state["messages"])

    question = st.chat_input("Ask a question about your documents...")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state["messages"].append(
        {"role": "user", "content": question, "sources": []}
    )

    if collection_count == 0:
        st.warning("Please upload documents first")
        return

    try:
        rag_chain = _build_or_get_chain(vectorstore, collection_count)
        result = query_rag(rag_chain, question)
        answer = str(result.get("answer", ""))
        sources = list(dict.fromkeys(result.get("sources", [])))

        with st.chat_message("assistant"):
            st.markdown(answer)
            if sources:
                _render_sources(sources)

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
    except Exception as exc:
        logger.exception("Query execution failed.")
        st.error(f"Error: {exc}")


if __name__ == "__main__":
    main()
