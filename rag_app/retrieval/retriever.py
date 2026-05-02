"""Retriever setup utilities for query-time vector search."""

from pathlib import Path

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from loguru import logger

from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    TOP_K,
)


def get_vectorstore() -> Chroma:
    """Create a LangChain Chroma vectorstore bound to local persisted storage.

    Returns:
        A configured Chroma vectorstore instance.

    Raises:
        RuntimeError: If vectorstore creation fails.
    """
    try:
        Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )
        logger.info(
            "Initialized vectorstore collection='{}' persist_directory='{}'",
            COLLECTION_NAME,
            CHROMA_PERSIST_DIR,
        )
        return vectorstore
    except Exception as exc:
        logger.exception("Failed to initialize Chroma vectorstore.")
        raise RuntimeError(
            "Could not reach Ollama. Make sure `ollama serve` is running at http://localhost:11434"
        ) from exc


def get_retriever(vectorstore: Chroma, top_k: int = TOP_K) -> VectorStoreRetriever:
    """Create an MMR retriever for diversified context retrieval.

    Args:
        vectorstore: Source Chroma vectorstore.
        top_k: Number of chunks to return after retrieval.

    Returns:
        A VectorStoreRetriever configured for MMR search.
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": top_k * 3},
    )
    logger.info("Created MMR retriever with k={} fetch_k={}", top_k, top_k * 3)
    return retriever
