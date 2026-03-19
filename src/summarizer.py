import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Teacher model — large, slower, more detailed
TEACHER_MODEL = "llama-3.3-70b-versatile"

# Student model — small, faster, efficient
STUDENT_MODEL = "llama-3.1-8b-instant"


def summarize(query: str, context_chunks: list[str], model: str) -> str:
    """
    Takes retrieved chunks as context and generates a summary
    using the specified LLM model (teacher or student).
    """
    context = "\n\n".join(context_chunks)

    prompt = f"""You are a news summarization assistant.
Using ONLY the news context provided below, write a clear and concise summary
that answers the user's query. Do not use any outside knowledge.

USER QUERY: {query}

NEWS CONTEXT:
{context}

Write a focused summary in 4-5 sentences:"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()


def summarize_teacher(query: str, context_chunks: list[str]) -> str:
    """Teacher LLM — high quality, slower (70B params)"""
    print(f"Teacher ({TEACHER_MODEL}) generating summary...")
    return summarize(query, context_chunks, TEACHER_MODEL)


def summarize_student(query: str, context_chunks: list[str]) -> str:
    """Student LLM — fast, efficient (8B params)"""
    print(f"Student ({STUDENT_MODEL}) generating summary...")
    return summarize(query, context_chunks, STUDENT_MODEL)


if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from news_fetcher import fetch_news
    from embedder import chunk_articles, store_embeddings
    from retriever import retrieve_relevant_chunks

    # Full pipeline test
    print("=== Fetching news... ===")
    articles = fetch_news("climate change", num_articles=5)
    chunks = chunk_articles(articles)
    store_embeddings(chunks)

    print("\n=== Retrieving context... ===")
    query = "How is climate change affecting animals?"
    context = retrieve_relevant_chunks(query, k=3)

    print("\n=== TEACHER SUMMARY (70B) ===")
    teacher_summary = summarize_teacher(query, context)
    print(teacher_summary)

    print("\n=== STUDENT SUMMARY (8B) ===")
    student_summary = summarize_student(query, context)
    print(student_summary)

    print("\n=== COMPARISON ===")
    print(f"Teacher length: {len(teacher_summary.split())} words")
    print(f"Student length: {len(student_summary.split())} words")