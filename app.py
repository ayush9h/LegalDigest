import re

import pdfplumber
import streamlit as st
from groq import Groq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# --------------------- Load Local FLAN-T5 Model -------------------------
@st.cache_resource
def load_local_model():
    tokenizer = AutoTokenizer.from_pretrained("./flan-t5-small-legal-finetuned")
    model = AutoModelForSeq2SeqLM.from_pretrained("./flan-t5-small-legal-finetuned")
    return tokenizer, model


tokenizer, local_model = load_local_model()


# ------------------- Groq-based Full Document Summarizer ----------------
def summarize_with_groq(text: str, api_key: str):
    client = Groq(api_key=api_key)
    prompt = f"Summarize the following legal document into one concise paragraph:\n\n{text[:4000]}"

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


# ------------------- Article Extraction ---------------------
def extract_articles(text):
    articles = re.split(r"\b(?:Article|Section)\s+\d+\s*[:\-]?", text)
    headers = re.findall(r"\b(?:Article|Section)\s+\d+\s*[:\-]?", text)
    result = []
    for i in range(1, len(articles)):
        result.append((headers[i - 1].strip(), articles[i].strip()))
    return result


# ------------------ Simple Crime Classification -------------
def classify_crime(article_text):
    crime_keywords = {
        "murder": ["kill", "homicide", "murder"],
        "theft": ["steal", "theft", "rob"],
        "fraud": ["fraud", "cheat", "mislead"],
        "assault": ["attack", "assault", "harm"],
    }
    for crime, keywords in crime_keywords.items():
        if any(word in article_text.lower() for word in keywords):
            return crime
    return "other"


# ------------------ Local Model Summarizer ------------------
def summarize_with_local_model(text, max_input_length=512, max_output_length=128):
    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    outputs = local_model.generate(inputs["input_ids"], max_length=max_output_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ------------------ Streamlit UI ----------------------------
st.title("Legal PDF Analyzer")

groq_api_key = st.text_input("Enter your Groq API Key", type="password")

pdf_file = st.file_uploader("Upload a Legal PDF", type=["pdf"])

if pdf_file and groq_api_key:
    with pdfplumber.open(pdf_file) as pdf:
        full_text = "\n".join(
            [page.extract_text() for page in pdf.pages if page.extract_text()]
        )

    st.subheader("Full Document Summary (via Groq LLM)")
    try:
        summary = summarize_with_groq(full_text, api_key=groq_api_key)
        st.write(summary)
    except Exception as e:
        st.error(f"Error summarizing with Groq: {str(e)}")

    st.subheader("Articles Grouped by Crime Type")
    articles = extract_articles(full_text)

    crime_to_articles = {}
    for header, content in articles:
        crime = classify_crime(content)
        summary = summarize_with_local_model(content[:1000])
        crime_to_articles.setdefault(crime, []).append((header, summary))

    for crime, items in crime_to_articles.items():
        with st.expander(
            f"Crime: {crime.title()} ({len(items)} article{'s' if len(items) > 1 else ''})"
        ):
            for header, summary in items:
                st.markdown(f"**{header}**")
                st.write(summary)
                st.markdown(f"**{header}**")
                st.write(summary)
