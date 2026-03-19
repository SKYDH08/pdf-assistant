import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma_db"

def get_retriever():
    """
    Loads ChromaDB and returns a retriever object.
    """
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedder
    )
    return vectorstore


def retrieve_relevant_chunks(query: str, k: int = 3) -> list[str]:
    """
    Given a user query, finds the k most semantically
    similar chunks from ChromaDB using cosine similarity.
    """
    vectorstore = get_retriever()

    results = vectorstore.similarity_search_with_score(query, k=k)

    print(f"\nTop {k} chunks retrieved for query: '{query}'")
    chunks = []
    for i, (doc, score) in enumerate(results):
        print(f"  Chunk {i+1} | Cosine score: {score:.4f}")
        chunks.append(doc.page_content)

    return chunks


if __name__ == "__main__":
    # First embed some articles, then test retrieval
    import sys
    sys.path.append("src")
    from news_fetcher import fetch_news
    from embedder import chunk_articles, store_embeddings

    # Fetch and store
    articles = fetch_news("climate change", num_articles=5)
    chunks = chunk_articles(articles)
    store_embeddings(chunks)

    # Now retrieve
    results = retrieve_relevant_chunks("global warming effects on animals", k=3)

    print("\n--- Retrieved Context ---")
    for i, chunk in enumerate(results):
        print(f"\nChunk {i+1}:\n{chunk[:300]}...")