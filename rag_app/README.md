# RAG App (LangChain + ChromaDB + Streamlit)

This project is a production-oriented Retrieval-Augmented Generation (RAG) application in Python 3.11 that indexes local documents into ChromaDB and answers user questions with grounded context using OpenAI embeddings and `gpt-4o-mini`.

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
├── .env.example
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

3. Configure environment variables:

   ```bash
   copy .env.example .env
   ```

   Then edit `.env` and set:
   ```env
   OPENAI_API_KEY=your_real_key
   ```

## Run Ingestion

After implementing the ingestion pipeline logic, run the ingestion module from this directory:

```bash
python -m ingestion.embedder
```

## Run the Streamlit App

```bash
streamlit run app.py
```

## RAG Architecture

The pipeline follows a standard local-first RAG flow: documents (`.pdf`, `.txt`, `.docx`) are loaded and normalized, split into overlapping chunks (`512/64`), embedded using `text-embedding-3-small`, and persisted in a ChromaDB collection (`rag_documents`) with on-disk storage. At query time, a retriever fetches top-k relevant chunks, an optional re-ranker narrows the context, and `gpt-4o-mini` (temperature `0`) generates a grounded answer using only retrieved evidence.
