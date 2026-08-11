"""
generate.py — Full RAG pipeline: retrieve relevant chunks from Redis,
then ask gemma3:1b to answer the question using only that retrieved text.

This is the first version WITHOUT conversation memory or caching —
those get added in later steps.
"""

import json
import redis
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

REDIS_URL = "redis://localhost:6379"
INDEX_NAME = "mybrain"
CACHE_INDEX_NAME = "mybrain-cache"
TOP_K = 4
HISTORY_TURNS = 5          # how many past turns to include as context
HISTORY_TTL_SECONDS = 3600 # session memory expires after 1 hour of inactivity
CACHE_SIMILARITY_THRESHOLD = 0.05  # lower = stricter match required (COSINE distance)
CACHE_TTL_SECONDS = 86400          # cached answers expire after 24 hours

PROMPT = ChatPromptTemplate.from_template(
    """You are an assistant answering questions using only the context below,
which comes from the user's own notes and documents.

Rules:
- Only use information from the context. Do not use outside knowledge.
- If the context doesn't contain the answer, say so plainly — don't guess.
- Use the conversation history to understand follow-up questions
  (e.g. "what about the second one?"), but still answer only from the context.
- Keep the answer concise.
- After the answer, list the source file(s) you used.

Conversation history:
{history}

Context:
{context}

Question: {question}

Answer:"""
)


def get_redis_client():
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


def load_history(r, session_id: str):
    raw = r.get(f"session:{session_id}")
    return json.loads(raw) if raw else []


def save_turn(r, session_id: str, question: str, answer: str):
    history = load_history(r, session_id)
    history.append({"question": question, "answer": answer})
    history = history[-HISTORY_TURNS:]  # keep only the most recent turns
    r.set(f"session:{session_id}", json.dumps(history), ex=HISTORY_TTL_SECONDS)


def format_history(history):
    if not history:
        return "(no previous turns)"
    lines = []
    for turn in history:
        lines.append(f"User: {turn['question']}\nAssistant: {turn['answer']}")
    return "\n\n".join(lines)


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


def get_cache_store(embeddings):
    """A separate Redis vector index just for cached question/answer pairs."""
    config = RedisConfig(
        index_name=CACHE_INDEX_NAME,
        redis_url=REDIS_URL,
        distance_metric="COSINE",
        metadata_schema=[
            {"name": "answer", "type": "text"},
            {"name": "sources", "type": "text"},
        ],
    )
    return RedisVectorStore(embeddings=embeddings, config=config)


def check_cache(cache_store, question: str):
    """Look for a semantically similar cached question. Returns
    (answer, sources) if found, else None."""
    results = cache_store.similarity_search_with_score(question, k=1)
    if not results:
        return None

    doc, score = results[0]
    if score <= CACHE_SIMILARITY_THRESHOLD:
        answer = doc.metadata.get("answer", "")
        sources = doc.metadata.get("sources", "")
        sources_list = sources.split(",") if sources else []
        return answer, sources_list
    return None


def save_to_cache(cache_store, question: str, answer: str, sources: list):
    cache_store.add_documents([
        Document(
            page_content=question,
            metadata={"answer": answer, "sources": ",".join(sources)},
        )
    ])


def build_context(results):
    """Turn retrieved (doc, score) pairs into a single context string,
    with each chunk labeled by its source file."""
    parts = []
    for doc, _score in results:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(vector_store, cache_store, llm, question: str, history: list):
    # 1. Check the semantic cache first
    cached = check_cache(cache_store, question)
    if cached is not None:
        answer, sources = cached
        return answer, sources, True  # True = served from cache

    # 2. No cache hit — do the real retrieval + generation
    results = vector_store.similarity_search_with_score(question, k=TOP_K)

    if not results:
        return "No relevant content found in your notes for this question.", [], False

    context = build_context(results)
    sources = sorted({doc.metadata.get("source", "unknown") for doc, _ in results})

    chain = PROMPT | llm
    response = chain.invoke({
        "context": context,
        "question": question,
        "history": format_history(history),
    })

    answer = response.content
    save_to_cache(cache_store, question, answer, sources)

    return answer, sources, False


def main():
    vector_store = get_vector_store()
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    cache_store = get_cache_store(embeddings)
    llm = ChatOllama(model="gemma3:1b")
    r = get_redis_client()

    # A fresh session each time you start the script. Later, the Streamlit
    # app will assign one session_id per browser session instead.
    session_id = "cli-session"
    r.delete(f"session:{session_id}")  # start clean each run

    print("Ask a question about your notes (or 'quit' to exit).\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        history = load_history(r, session_id)
        answer, sources, from_cache = answer_question(
            vector_store, cache_store, llm, question, history
        )
        save_turn(r, session_id, question, answer)

        tag = " (from cache)" if from_cache else ""
        print(f"\nAssistant{tag}: {answer}")
        if sources:
            print(f"\n(Sources: {', '.join(sources)})")
        print()


if __name__ == "__main__":
    main()