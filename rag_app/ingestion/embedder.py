"""Embedding and vectorstore upsert logic for the ingestion pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from loguru import logger

from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL, OLLAMA_BASE_URL
from ingestion.chunker import chunk_documents
from ingestion.loader import load_document


def get_chroma_client() -> Chroma:
    """Create a LangChain Chroma vectorstore for local persistent storage.

    Returns:
        A Chroma vectorstore configured for local Ollama embeddings.

    Raises:
        RuntimeError: If vectorstore initialization fails.
    """
    try:
        persist_dir = Path(CHROMA_PERSIST_DIR)
        persist_dir.mkdir(parents=True, exist_ok=True)
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )
        logger.info("Initialized Chroma vectorstore at '{}'", persist_dir)
        return vectorstore
    except Exception as exc:
        logger.exception("Failed to initialize Chroma vectorstore.")
        raise RuntimeError(
            "Could not reach Ollama. Make sure `ollama serve` is running at http://localhost:11434"
        ) from exc


def get_or_create_collection(client: Chroma) -> Chroma:
    """Return the configured Chroma vectorstore collection wrapper.

    Args:
        client: Initialized Chroma vectorstore.

    Returns:
        The same Chroma vectorstore instance.

    Raises:
        RuntimeError: If the collection cannot be accessed.
    """
    try:
        _ = client._collection
        logger.info("Using Chroma collection '{}'", COLLECTION_NAME)
        return client
    except Exception as exc:
        logger.exception("Failed accessing Chroma collection.")
        raise RuntimeError(
            "Could not reach Ollama. Make sure `ollama serve` is running at http://localhost:11434"
        ) from exc


def embed_and_store(chunks: list[Document], collection: Chroma) -> int:
    """Embed chunks and add only unseen IDs into Chroma.

    Args:
        chunks: Chunked documents to embed and persist.
        collection: Target Chroma vectorstore.

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
            collection.add_documents(
                documents=[chunk for _, chunk in batch],
                ids=[chunk_id for chunk_id, _ in batch],
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
        raise RuntimeError(
            "Could not reach Ollama. Make sure `ollama serve` is running at http://localhost:11434"
        ) from exc


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
        raise RuntimeError(
            "Could not reach Ollama. Make sure `ollama serve` is running at http://localhost:11434"
        ) from exc


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
