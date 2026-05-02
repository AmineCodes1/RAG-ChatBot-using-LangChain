"""Central configuration for the local Ollama-based RAG application."""

from __future__ import annotations

import requests
from loguru import logger

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "rag_documents"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 5
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 1024
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx"]


def check_ollama_running() -> bool:
    """Check whether the local Ollama service is reachable."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return response.status_code == 200
    except Exception:
        logger.warning("Ollama is not reachable at {}", OLLAMA_BASE_URL)
        return False
