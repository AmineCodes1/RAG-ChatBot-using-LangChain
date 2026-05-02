"""RAG chain and query helpers for retrieval-augmented answering."""

from typing import Any

from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from loguru import logger

from config import LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE, OLLAMA_BASE_URL


def build_rag_chain(retriever: VectorStoreRetriever) -> RetrievalQA:
    """Build a RetrievalQA chain with a strict context-grounded prompt.

    Args:
        retriever: Configured retriever used to fetch context documents.

    Returns:
        A RetrievalQA chain returning both answer and source documents.

    Raises:
        RuntimeError: If chain creation fails.
    """
    try:
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful assistant. Answer ONLY using the provided context.\n"
                "If the answer is not in the context, say "
                "'I don't have enough information to answer this.'\n"
                "Never make up information.\n\n"
                "Context: {context}\n"
                "Question: {question}\n"
                "Answer:"
            ),
        )
        llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
            num_predict=LLM_MAX_TOKENS,
        )
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        logger.info(
            "Built RetrievalQA chain model='{}' temperature={} max_tokens={}",
            LLM_MODEL,
            LLM_TEMPERATURE,
            LLM_MAX_TOKENS,
        )
        return chain
    except Exception as exc:
        logger.exception("Failed to build RetrievalQA chain.")
        raise RuntimeError(
            "Could not reach Ollama. Make sure `ollama serve` is running at http://localhost:11434"
        ) from exc


def query_rag(chain: RetrievalQA, question: str) -> dict[str, Any]:
    """Run a question against the RAG chain and normalize output shape.

    Args:
        chain: Configured RetrievalQA chain.
        question: User question to answer.

    Returns:
        A dictionary with the generated answer and unique source names.
    """
    try:
        response = chain.invoke({"query": question})
        answer = str(response.get("result", ""))
        source_documents = response.get("source_documents", [])

        unique_sources: list[str] = []
        seen_sources: set[str] = set()
        for document in source_documents:
            source = str((document.metadata or {}).get("source", "")).strip()
            if source and source not in seen_sources:
                seen_sources.add(source)
                unique_sources.append(source)

        logger.info(
            "RAG query question='{}' answer_length={} sources={}",
            question[:80],
            len(answer),
            len(unique_sources),
        )
        return {"answer": answer, "sources": unique_sources}
    except Exception as exc:
        logger.exception("RAG query failed.")
        return {
            "answer": (
                "Error: Could not reach Ollama. Make sure `ollama serve` is running "
                "at http://localhost:11434"
            ),
            "sources": [],
        }
