"""
query.py — Test retrieval only (no LLM yet).

Takes a question, embeds it with nomic-embed-text, and searches
the Redis vector index for the most similar chunks. This lets you
sanity-check retrieval quality before adding generation on top.
"""

from langchain_ollama import OllamaEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore

REDIS_URL = "redis://localhost:6379"
INDEX_NAME = "mybrain"


def get_vector_store():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    config = RedisConfig(
        index_name=INDEX_NAME,
        redis_url=REDIS_URL,
        distance_metric="COSINE",
        metadata_schema=[
            {"name": "source", "type": "tag"},
        ],
    )
    return RedisVectorStore(embeddings=embeddings, config=config)


def search(vector_store, question: str, k: int = 4):
    """Return the top-k most similar chunks, with similarity scores."""
    results = vector_store.similarity_search_with_score(question, k=k)
    return results


def main():
    vector_store = get_vector_store()

    print("Type a question to test retrieval (or 'quit' to exit).\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        results = search(vector_store, question)

        if not results:
            print("No results found. Did you run ingest.py first?\n")
            continue

        print(f"\nTop {len(results)} matches:\n")
        for i, (doc, score) in enumerate(results, start=1):
            source = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:200].replace("\n", " ")
            print(f"[{i}] score={score:.4f} | source={source}")
            print(f"    {preview}...\n")


if __name__ == "__main__":
    main()