"""Text chunking strategies for preparing documents for vector indexing."""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger

from config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_documents(
    docs: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split loaded documents into retrieval-ready chunks.

    Args:
        docs: Input documents to split.
        chunk_size: Maximum chunk size to generate.
        chunk_overlap: Character overlap between adjacent chunks.

    Returns:
        A list of chunked Document objects with preserved metadata.

    Raises:
        RuntimeError: If chunking fails.
    """
    try:
        logger.info(
            "Starting chunking for input_docs={} chunk_size={} chunk_overlap={}",
            len(docs),
            chunk_size,
            chunk_overlap,
        )
        if not docs:
            logger.info("No input documents provided; returning empty chunk list.")
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        chunked_documents: list[Document] = []
        for source_document in docs:
            source_chunks = splitter.split_documents([source_document])
            source_metadata = dict(source_document.metadata or {})
            for chunk_index, chunk in enumerate(source_chunks):
                chunk.metadata = {**source_metadata, "chunk_index": chunk_index}
                chunked_documents.append(chunk)

        average_chunk_size = (
            sum(len(chunk.page_content) for chunk in chunked_documents)
            / len(chunked_documents)
            if chunked_documents
            else 0.0
        )
        logger.info(
            "Chunking complete input_docs={} output_chunks={} avg_chunk_size_chars={:.2f}",
            len(docs),
            len(chunked_documents),
            average_chunk_size,
        )
        return chunked_documents
    except Exception as exc:
        logger.exception("Failed chunking documents.")
        raise RuntimeError("Failed to chunk input documents.") from exc
