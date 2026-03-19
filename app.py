import streamlit as st
import sys
import os

sys.path.append("src")

from news_fetcher import fetch_news
from embedder import chunk_articles, store_embeddings
from retriever import retrieve_relevant_chunks
from summarizer import summarize_teacher, summarize_student
from translator import translate_all

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Multilingual News Summarizer",
    page_icon="🌍",
    layout="wide"
)

# ─── Header ────────────────────────────────────────────────
st.title("🌍 Multilingual News Summarizer")
st.markdown("**RAG + Knowledge Distillation + Multilingual Translation**")
st.divider()

# ─── Sidebar Controls ──────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    topic = st.text_input(
        "News Topic",
        value="artificial intelligence",
        help="Any topic you want to search news for"
    )

    num_articles = st.slider(
        "Number of articles to fetch",
        min_value=3,
        max_value=15,
        value=5
    )

    languages = st.multiselect(
        "Output Languages",
        options=["English", "Hindi", "French", "Spanish"],
        default=["English", "Hindi"]
    )

    show_kd = st.checkbox(
        "Show Knowledge Distillation comparison",
        value=True
    )

    fetch_btn = st.button("🔍 Fetch & Summarize", type="primary", use_container_width=True)

    st.divider()
    st.markdown("### 📖 How it works")
    st.markdown("""
    1. **Fetch** — NewsAPI pulls latest articles
    2. **Embed** — Text converted to vectors
    3. **Store** — ChromaDB indexes chunks
    4. **Retrieve** — Cosine similarity search
    5. **Summarize** — LLM generates summary
    6. **Translate** — Helsinki-NLP translates
    """)

# ─── Main Flow ─────────────────────────────────────────────
if fetch_btn:
    if not topic:
        st.error("Please enter a news topic!")
        st.stop()

    if not languages:
        st.error("Please select at least one language!")
        st.stop()

    # Step 1: Fetch news
    with st.status("🔍 Fetching latest news...", expanded=True) as status:
        st.write(f"Searching for: **{topic}**")
        articles = fetch_news(topic, num_articles=num_articles)

        if not articles:
            status.update(label="❌ Failed to fetch news", state="error")
            st.error("Could not fetch articles. Check your NEWS_API_KEY.")
            st.stop()

        st.write(f"✅ Fetched **{len(articles)}** articles")

        # Step 2: Chunk and embed
        st.write("Chunking and embedding articles...")
        chunks = chunk_articles(articles)
        store_embeddings(chunks)
        st.write(f"✅ Created **{len(chunks)}** chunks, stored in ChromaDB")

        # Step 3: Retrieve
        st.write("Retrieving relevant context...")
        context = retrieve_relevant_chunks(topic, k=3)
        st.write(f"✅ Retrieved **{len(context)}** relevant chunks")

        status.update(label="✅ Pipeline complete!", state="complete")

    st.divider()

    # Step 4: Show fetched articles
    with st.expander(f"📰 Fetched Articles ({len(articles)})", expanded=False):
        for i, article in enumerate(articles):
            st.markdown(f"**{i+1}. [{article['title']}]({article['url']})**")
            st.caption(f"{article['source']} • {article['published'][:10]}")
            st.write(article['description'])
            st.divider()

    # Step 5: Knowledge Distillation comparison
    if show_kd:
        st.subheader("🧠 Knowledge Distillation — Teacher vs Student")
        st.caption("Same context, two different sized models. See how quality compares.")

        col1, col2 = st.columns(2)

        with col1:
            with st.spinner("Teacher LLM (70B) thinking..."):
                teacher_summary = summarize_teacher(topic, context)
            st.markdown("### 👨‍🏫 Teacher LLM")
            st.caption("Llama 3.3 70B — Large, detailed, slower")
            st.info(teacher_summary)
            st.metric("Word count", len(teacher_summary.split()))

        with col2:
            with st.spinner("Student LLM (8B) thinking..."):
                student_summary = summarize_student(topic, context)
            st.markdown("### 👨‍🎓 Student LLM")
            st.caption("Llama 3.1 8B — Compact, fast, efficient")
            st.success(student_summary)
            st.metric("Word count", len(student_summary.split()))

    else:
        # Just use student model silently
        with st.spinner("Generating summary..."):
            student_summary = summarize_student(topic, context)

    st.divider()

    # Step 6: Multilingual output
    st.subheader("🌐 Multilingual Translations")

    if "English" not in languages:
        languages = ["English"] + languages

    with st.spinner("Translating to selected languages..."):
        translations = translate_all(student_summary, languages)

    lang_flags = {
        "English": "🇬🇧",
        "Hindi":   "🇮🇳",
        "French":  "🇫🇷",
        "Spanish": "🇪🇸",
    }

    for lang, text in translations.items():
        flag = lang_flags.get(lang, "🌐")
        with st.expander(f"{flag} {lang}", expanded=True):
            st.write(text)

    st.divider()

    # Step 7: Retrieved context (transparency)
    with st.expander("🔍 Retrieved Context Chunks (RAG)", expanded=False):
        st.caption("These are the actual news chunks passed to the LLM as context")
        for i, chunk in enumerate(context):
            st.markdown(f"**Chunk {i+1}:**")
            st.text(chunk[:400] + "...")
            st.divider()

else:
    # Landing state
    st.info("👈 Enter a topic in the sidebar and click **Fetch & Summarize** to begin.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Components Built", "5/6", "83%")
    with col2:
        st.metric("Languages Supported", "4", "+English")
    with col3:
        st.metric("Models Used", "2", "Teacher + Student")