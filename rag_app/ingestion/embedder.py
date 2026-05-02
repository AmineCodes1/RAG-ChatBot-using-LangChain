"""Embedding and ChromaDB upsert logic for the ingestion pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_core.documents import Document
from loguru import logger

from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)
from ingestion.chunker import chunk_documents
from ingestion.loader import load_document


def get_chroma_client() -> chromadb.PersistentClient:
    """Create a persistent ChromaDB client for local vector storage.

    Returns:
        A ChromaDB PersistentClient configured with the local persistence path.

    Raises:
        RuntimeError: If client initialization fails.
    """
    try:
        persist_dir = Path(CHROMA_PERSIST_DIR)
        persist_dir.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=str(persist_dir))
        logger.info("Initialized Chroma persistent client at '{}'", persist_dir)
        return client
    except Exception as exc:
        logger.exception("Failed to initialize Chroma client.")
        raise RuntimeError("Failed to initialize ChromaDB persistent client.") from exc


def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Get or create the configured ChromaDB collection.

    Args:
        client: Initialized ChromaDB PersistentClient.

    Returns:
        The requested ChromaDB collection with OpenAI embedding function.

    Raises:
        RuntimeError: If collection creation or retrieval fails.
    """
    try:
        embedding_function = OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=EMBEDDING_MODEL,
        )
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
        )
        logger.info("Using Chroma collection '{}'", COLLECTION_NAME)
        return collection
    except Exception as exc:
        logger.exception("Failed getting or creating Chroma collection.")
        raise RuntimeError("Failed to get or create Chroma collection.") from exc


def embed_and_store(chunks: list[Document], collection: chromadb.Collection) -> int:
    """Embed chunks and upsert only unseen chunk IDs into ChromaDB.

    Args:
        chunks: Chunked documents to embed and store.
        collection: Target ChromaDB collection.

    Returns:
        Number of newly added chunks.

    Raises:
        RuntimeError: If embedding or storage fails.
    """
    try:
        if not chunks:
            logger.info("No chunks provided for embedding; nothing to store.")
            return 0

        unique_records: list[tuple[str, Document]] = []
        seen_ids: set[str] = set()
        skipped_duplicate_input = 0

        for chunk in chunks:
            chunk_id = _build_chunk_id(chunk)
            if chunk_id in seen_ids:
                skipped_duplicate_input += 1
                continue
            seen_ids.add(chunk_id)
            unique_records.append((chunk_id, chunk))

        unique_ids = [chunk_id for chunk_id, _ in unique_records]
        existing_response = collection.get(ids=unique_ids)
        existing_ids = _extract_existing_ids(existing_response.get("ids", []))

        records_to_add = [
            (chunk_id, chunk)
            for chunk_id, chunk in unique_records
            if chunk_id not in existing_ids
        ]

        for batch_start in range(0, len(records_to_add), 100):
            batch = records_to_add[batch_start : batch_start + 100]
            collection.upsert(
                ids=[chunk_id for chunk_id, _ in batch],
                documents=[chunk.page_content for _, chunk in batch],
                metadatas=[chunk.metadata for _, chunk in batch],
            )

        logger.info(
            (
                "Embedding complete total_chunks={} added={} skipped_existing={} "
                "skipped_duplicate_input={}"
            ),
            len(chunks),
            len(records_to_add),
            len(existing_ids),
            skipped_duplicate_input,
        )
        return len(records_to_add)
    except Exception as exc:
        logger.exception("Failed embedding and storing chunks.")
        raise RuntimeError("Failed to embed and store chunks in ChromaDB.") from exc


def run_ingestion_pipeline(file_paths: list[str]) -> dict[str, int]:
    """Run the end-to-end ingestion pipeline for a set of file paths.

    Args:
        file_paths: Paths of files to load, chunk, and index.

    Returns:
        Pipeline summary with files processed and chunk indexing counts.

    Raises:
        RuntimeError: If a critical pipeline error occurs.
    """
    try:
        logger.info("Starting ingestion pipeline for files={}", len(file_paths))

        all_documents: list[Document] = []
        failed_files: list[str] = []
        for file_path in file_paths:
            try:
                all_documents.extend(load_document(file_path))
            except Exception as exc:
                failed_files.append(file_path)
                logger.error("Failed to load file='{}': {}", file_path, exc)

        if failed_files:
            logger.warning("Files failed during loading: {}", failed_files)

        if not all_documents:
            result = {
                "files_processed": len(file_paths),
                "chunks_added": 0,
                "chunks_skipped": 0,
            }
            logger.warning("No documents loaded. Returning result={}", result)
            return result

        chunks = chunk_documents(all_documents)
        client = get_chroma_client()
        collection = get_or_create_collection(client)
        chunks_added = embed_and_store(chunks, collection)
        chunks_skipped = max(len(chunks) - chunks_added, 0)

        result = {
            "files_processed": len(file_paths),
            "chunks_added": chunks_added,
            "chunks_skipped": chunks_skipped,
        }
        logger.info("Ingestion pipeline complete result={}", result)
        return result
    except Exception as exc:
        logger.exception("Critical ingestion pipeline failure.")
        raise RuntimeError("Ingestion pipeline failed due to a critical error.") from exc


def _build_chunk_id(chunk: Document) -> str:
    """Build a stable chunk identifier from metadata and chunk content."""
    source = str((chunk.metadata or {}).get("source", ""))
    chunk_index = str((chunk.metadata or {}).get("chunk_index", ""))
    content_prefix = chunk.page_content[:100]
    raw_identifier = f"{source}{chunk_index}{content_prefix}"
    return hashlib.md5(raw_identifier.encode("utf-8")).hexdigest()


def _extract_existing_ids(raw_ids: Any) -> set[str]:
    """Normalize Chroma get() ID payloads to a flat set of IDs."""
    if not isinstance(raw_ids, list):
        return set()

    normalized_ids: set[str] = set()
    for item in raw_ids:
        if isinstance(item, list):
            normalized_ids.update(str(inner) for inner in item)
        else:
            normalized_ids.add(str(item))
    return normalized_ids
