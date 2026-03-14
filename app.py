import streamlit as st
from google import genai
import fitz  
import os
from dotenv import load_dotenv

load_dotenv()

# New google-genai client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_text_from_pdf(uploaded_file):
    """Extract all text from a PDF file."""
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ---------- UI ----------
st.set_page_config(page_title="📄 PDF Assistant", layout="centered")
st.title("📄 Document-Aware Assistant")
st.write("Upload a PDF and ask questions about it!")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
    st.success(f"✅ PDF loaded! ({len(pdf_text)} characters extracted)")

    question = st.text_input("Ask a question about the document")

    if question:
        with st.spinner("Thinking..."):
            prompt = f"""You are a helpful assistant.
Answer the question based ONLY on the document content below.
If the answer is not in the document, say "I couldn't find that in the document."

Document:
{pdf_text[:10000]}

Question: {question}
"""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

        st.write("### 💬 Answer")
        st.write(response.text)
