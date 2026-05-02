"""General helper utilities shared across the RAG application."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from loguru import logger


def sanitize_question(question: str) -> str:
    """Normalize and validate a user question.

    Args:
        question: Raw user input question.

    Returns:
        A sanitized question with collapsed internal whitespace.

    Raises:
        ValueError: If the question is empty after sanitization or exceeds 1000 chars.
    """
    sanitized_question = re.sub(r"\s+", " ", question.strip())
    if not sanitized_question:
        raise ValueError("Question cannot be empty.")
    if len(sanitized_question) > 1000:
        raise ValueError("Question cannot be longer than 1000 characters.")
    return sanitized_question


def format_sources(sources: list[str]) -> str:
    """Format unique, sorted source names for markdown display.

    Args:
        sources: Source filenames collected from retrieval metadata.

    Returns:
        A markdown-formatted source line or an empty string if no sources exist.
    """
    cleaned_sources = {source.strip() for source in sources if source and source.strip()}
    if not cleaned_sources:
        return ""
    sorted_sources = sorted(cleaned_sources, key=str.lower)
    return f"**Sources:** {', '.join(sorted_sources)}"


def get_collection_stats(collection: Any) -> dict[str, int | str]:
    """Get basic collection statistics.

    Args:
        collection: Chroma collection instance or None.

    Returns:
        A dictionary containing document count and collection name.
    """
    if collection is None:
        return {"document_count": 0, "collection_name": ""}

    document_count = int(collection.count())
    collection_name = str(getattr(collection, "name", ""))
    return {"document_count": document_count, "collection_name": collection_name}


def delete_chroma_db(persist_dir: str) -> bool:
    """Delete a Chroma persistence directory if it exists.

    Args:
        persist_dir: Path to the Chroma persistence directory.

    Returns:
        True if the directory was deleted, False if it did not exist.
    """
    target_dir = Path(persist_dir)
    if not target_dir.exists():
        logger.info("Chroma directory not found, nothing to delete: '{}'", target_dir)
        return False

    shutil.rmtree(target_dir)
    logger.warning("Deleted Chroma persistence directory: '{}'", target_dir)
    return True
