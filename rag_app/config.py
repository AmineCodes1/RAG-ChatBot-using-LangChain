"""Central configuration for the RAG application."""

import os
from typing import Final

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a local .env file into process env.

OPENAI_API_KEY: Final[str] = os.environ.get("OPENAI_API_KEY", "").strip()  # OpenAI API key for embeddings and LLM calls.
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is missing. Set it in your environment or .env file."
    )

CHROMA_PERSIST_DIR: Final[str] = "./chroma_db"  # Local directory where ChromaDB stores persistent vector data.
COLLECTION_NAME: Final[str] = "rag_documents"  # ChromaDB collection name used for all indexed documents.

CHUNK_SIZE: Final[int] = 512  # Default max token/character window per text chunk for retrieval indexing.
CHUNK_OVERLAP: Final[int] = 64  # Default overlap between adjacent chunks to preserve context continuity.

TOP_K: Final[int] = 5  # Default number of documents to retrieve from the vector store per query.
RERANK_TOP_N: Final[int] = 3  # Default number of retrieved documents kept after optional re-ranking.

MODEL_NAME: Final[str] = "gpt-4o-mini"  # OpenAI chat model used to generate grounded answers.
TEMPERATURE: Final[float] = 0  # Deterministic decoding for factual RAG responses.
MAX_TOKENS: Final[int] = 1024  # Maximum completion length for answer generation.

EMBEDDING_MODEL: Final[str] = "text-embedding-3-small"  # OpenAI embedding model used for document/query vectors.

SUPPORTED_EXTENSIONS: Final[list[str]] = [
    ".pdf",
    ".txt",
    ".docx",
]  # Allowed file extensions for ingestion loaders.
