"""Streamlit entry point for the RAG application."""

import streamlit as st


def main() -> None:
    """Render the initial Streamlit application shell."""
    st.set_page_config(page_title="RAG App", page_icon="📚", layout="wide")
    st.title("RAG Application")
    st.info("Scaffold created. Ingestion and retrieval pipeline modules will be wired next.")


if __name__ == "__main__":
    main()
