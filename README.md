# RAG App (LangChain + ChromaDB + Streamlit + Ollama)

This project is a production-oriented Retrieval-Augmented Generation (RAG) application in Python 3.11 that indexes local documents into ChromaDB and answers questions with grounded context using fully local Ollama models (`llama3.2` + `nomic-embed-text`).

## Project Structure

```text
rag_app/
├── app.py
├── config.py
├── ingestion/
│   ├── __init__.py
│   ├── loader.py
│   ├── chunker.py
│   └── embedder.py
├── retrieval/
│   ├── __init__.py
│   ├── retriever.py
│   ├── reranker.py
│   └── chain.py
├── utils/
│   ├── __init__.py
│   └── helpers.py
├── tests/
│   └── smoke_test.py
└── requirements.txt
```

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   **PowerShell**
   ```bash
   .\.venv\Scripts\Activate.ps1
   ```

   **Command Prompt**
   ```bash
   .\.venv\Scripts\activate.bat
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Pull required Ollama models:

   ```bash
   ollama pull llama3.2
   ```

   ```bash
   ollama pull nomic-embed-text
   ```

4. Start Ollama:

   ```bash
   ollama serve
   ```

## Run Ingestion

After implementing the ingestion pipeline logic, run the ingestion module from this directory:

```bash
python -m ingestion.embedder
```

Or run the smoke test:

```bash
python tests/smoke_test.py
```

## Run the Streamlit App

```bash
streamlit run app.py
```

## RAG Architecture

The pipeline follows a local-first RAG flow: documents (`.pdf`, `.txt`, `.docx`) are loaded and normalized, split into overlapping chunks (`512/64`), embedded locally through Ollama with `nomic-embed-text`, and persisted in ChromaDB (`rag_documents`) on disk. At query time, an MMR retriever fetches top-k relevant chunks and `llama3.2` generates grounded answers using only retrieved context.

