"""
ingest.py — Loads documents from ./data, splits them into chunks,
embeds each chunk with Ollama (nomic-embed-text), and stores them
in Redis as a searchable vector index.

Run this whenever you add new files to the data/ folder.
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore

DATA_DIR = "data"
REDIS_URL = "redis://localhost:6379"
INDEX_NAME = "mybrain"


def load_documents(data_dir: str):
    """Load all PDFs, .txt, and .md files from the data folder."""
    documents = []

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)

        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            docs = loader.load()
        elif filename.lower().endswith((".txt", ".md")):
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
        else:
            print(f"Skipping unsupported file: {filename}")
            continue

        # Tag every chunk with its source filename so we can cite it later
        for doc in docs:
            doc.metadata["source"] = filename

        documents.extend(docs)
        print(f"Loaded {filename} ({len(docs)} page(s)/section(s))")

    return documents


def chunk_documents(documents):
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def main():
    if not os.path.isdir(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"No files found in ./{DATA_DIR}. Add PDFs, .txt, or .md files first.")
        return

    # 1. Load raw documents
    documents = load_documents(DATA_DIR)
    if not documents:
        print("No supported documents were loaded. Exiting.")
        return

    # 2. Split into chunks
    chunks = chunk_documents(documents)

    # 3. Set up the embedding model (local, via Ollama)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 4. Configure the Redis vector index
    config = RedisConfig(
        index_name=INDEX_NAME,
        redis_url=REDIS_URL,
        distance_metric="COSINE",
        metadata_schema=[
            {"name": "source", "type": "tag"},
        ],
    )

    # 5. Push chunks + embeddings into Redis, in small batches so we
    #    don't overwhelm the local Ollama embedding server in one request
    vector_store = RedisVectorStore(embeddings=embeddings, config=config)

    batch_size = 25
    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start:start + batch_size]
        vector_store.add_documents(batch)
        print(f"Embedded {min(start + batch_size, total)}/{total} chunks...")

    print(f"\nDone. {total} chunks stored in Redis index '{INDEX_NAME}'.")
    print("Open http://localhost:8001 (RedisInsight) to inspect the stored data.")


if __name__ == "__main__":
    main()