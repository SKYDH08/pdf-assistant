import requests
import os
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def fetch_news(topic: str, num_articles: int = 10) -> list[dict]:
    """
    Fetches news articles from NewsAPI based on a topic/keyword.
    Returns a list of dicts with title, description, content, url.
    """
    url = "https://newsapi.org/v2/everything"

    params = {
        "q": topic,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": num_articles,
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Error fetching news: {response.status_code} - {response.text}")
        return []

    data = response.json()
    articles = []

    for article in data.get("articles", []):
        # Skip articles with missing content
        if not article.get("content") or not article.get("title"):
            continue

        articles.append({
            "title":       article.get("title", ""),
            "description": article.get("description", ""),
            "content":     article.get("content", ""),
            "url":         article.get("url", ""),
            "source":      article.get("source", {}).get("name", "Unknown"),
            "published":   article.get("publishedAt", ""),
        })

    print(f"Fetched {len(articles)} articles for topic: '{topic}'")
    return articles


if __name__ == "__main__":
    # Quick test — run this file directly to verify it works
    articles = fetch_news("artificial intelligence", num_articles=5)
    for i, a in enumerate(articles):
        print(f"\n--- Article {i+1} ---")
        print(f"Title:   {a['title']}")
        print(f"Source:  {a['source']}")
        print(f"Content: {a['content'][:200]}...")