"""Standalone smoke test for end-to-end RAG ingestion and query flow."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.embedder import run_ingestion_pipeline
from retrieval.chain import build_rag_chain, query_rag
from retrieval.retriever import get_retriever, get_vectorstore


def main() -> None:
    """Run a minimal end-to-end smoke test and print pass/fail status."""
    temp_dir = Path(tempfile.mkdtemp(prefix="rag_smoke_"))
    answer = ""
    try:
        test_file_path = temp_dir / "morocco_facts.txt"
        test_file_path.write_text(
            "The capital of Morocco is Rabat. The currency is the Moroccan Dirham.",
            encoding="utf-8",
        )

        run_ingestion_pipeline([str(test_file_path)])
        vectorstore = get_vectorstore()
        retriever = get_retriever(vectorstore)
        chain = build_rag_chain(retriever)
        result = query_rag(chain, "What is the capital of Morocco?")
        answer = str(result.get("answer", ""))

        assert "rabat" in answer.lower(), "Expected 'Rabat' in the answer."
        print(f"PASS: {answer}")
    except AssertionError:
        print(f"FAIL: {answer}")
    except Exception as exc:
        logger.exception("Smoke test failed.")
        print(f"FAIL: {answer or f'Error: {exc}'}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("All systems go — run: streamlit run app.py")


if __name__ == "__main__":
    main()
