# AiRAG

A simple local retrieval-augmented generation (RAG) project for building a private document assistant.

## What it does

- Ingests documents from `data/` (`.pdf`, `.txt`, `.md`)
- Splits content into chunks for embedding
- Embeds text using Ollama's `nomic-embed-text`
- Stores vectors in Redis for semantic search
- Supports retrieval testing and question answering with a local Ollama chat model

## Files

- `ingest.py` — load files, split into chunks, embed them, and store vectors in Redis
- `query.py` — test retrieval by searching the Redis index with a question
- `generate.py` — run a full retriever + generator pipeline using `gemma3:1b`
- `app.py` — currently empty placeholder
- `requirements.txt` — Python dependencies

## Requirements

- Python 3.11+ (recommended)
- Redis running locally on `redis://localhost:6379`
- Ollama installed and accessible from your environment
- A `data/` folder containing files to ingest

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Redis locally and verify it is reachable.

4. Place your documents in the `data/` folder.

## Usage

### Ingest documents

```bash
python ingest.py
```

This loads supported files from `data/`, splits them into chunks, embeds them with Ollama, and stores them in Redis.

### Test retrieval only

```bash
python query.py
```

This lets you ask a question and see the top matching chunks from the Redis index.

### Ask questions with generation

```bash
python generate.py
```

This retrieves relevant chunks and uses `gemma3:1b` to answer the query based only on your document context.

## Notes

- The project uses Redis for vector storage and session history.
- `generate.py` includes a simple semantic cache and conversation history logic.
- `app.py` is currently unused and can be extended later.

## License

Use as you like as this was just a simple project that I made to learn a bit about RAG.
