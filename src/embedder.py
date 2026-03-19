import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "chroma_db"

def get_embedder():
    """
    Loads the sentence-transformer embedding model.
    Downloads on first run (~90MB), cached after that.
    """
    print("Loading embedding model...")
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Embedding model loaded.")
    return embedder


def chunk_articles(articles: list[dict]) -> list[str]:
    """
    Splits articles into smaller chunks for better retrieval.
    Each chunk is ~300 words with 50 word overlap.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = []
    for article in articles:
        # Combine title + description + content for richer context
        full_text = f"Title: {article['title']}\n\n{article['description']}\n\n{article['content']}"
        article_chunks = splitter.split_text(full_text)
        chunks.extend(article_chunks)

    print(f"Split {len(articles)} articles into {len(chunks)} chunks")
    return chunks


def store_embeddings(chunks: list[str]) -> Chroma:
    """
    Converts chunks to embeddings and stores in ChromaDB.
    ChromaDB runs fully locally — no server needed.
    """
    embedder = get_embedder()

    print("Storing embeddings in ChromaDB...")
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embedder,
        persist_directory=CHROMA_PATH
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB at '{CHROMA_PATH}'")
    return vectorstore


def load_vectorstore() -> Chroma:
    """
    Loads an existing ChromaDB vectorstore from disk.
    """
    embedder = get_embedder()
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedder
    )
    return vectorstore


if __name__ == "__main__":
    # Quick test — fetch some news and embed it
    from news_fetcher import fetch_news

    articles = fetch_news("climate change", num_articles=5)
    chunks = chunk_articles(articles)
    vectorstore = store_embeddings(chunks)
    print("\nEmbedding test complete!")
    print(f"Sample chunk:\n{chunks[0][:300]}")