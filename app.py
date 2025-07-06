import re

import pandas as pd
import pdfplumber
import streamlit as st
from groq import Groq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@st.cache_resource
def load_local_model():
    tokenizer = AutoTokenizer.from_pretrained("./flan-t5-small-legal-finetuned")
    model = AutoModelForSeq2SeqLM.from_pretrained("./flan-t5-small-legal-finetuned")
    return tokenizer, model

tokenizer, local_model = load_local_model()

def summarize_with_groq(text: str, api_key: str):
    client = Groq(api_key=api_key)
    prompt = f"Summarize the following legal document into one concise paragraph with important legal sections: add a keyword section for identification\n\n{text[:4000]}"
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()

def summarize_with_local_model(text, max_input_length=512, max_output_length=128):
    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    outputs = local_model.generate(inputs["input_ids"], max_length=max_output_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

st.set_page_config(page_title="Legal Digest", layout="wide")
st.title("Legal Digest")

groq_api_key = st.text_input("Enter your Groq API Key", type="password")
pdf_file = st.file_uploader("Upload a Legal PDF", type=["pdf"])

if pdf_file and groq_api_key:
    with pdfplumber.open(pdf_file) as pdf:
        full_text = "\n".join(
            [page.extract_text() for page in pdf.pages if page.extract_text()]
        )

    try:
        summarized_document = summarize_with_groq(full_text, api_key=groq_api_key)
        st.success("Document summarized successfully.")
    except Exception as e:
        st.error(f"Error summarizing with Groq: {str(e)}")
        summarized_document = ""

    summary_articles = summarize_with_local_model(summarized_document)

    ipc_sections = list(set(re.findall(r"Sections? (\d+[A-Z]?)", summarized_document)))
    ipc_df = pd.read_csv("./datasets/IPC.csv")
    ipc_df["SectionNumber"] = ipc_df["IPC_Section"].str.extract(r"(\d+[A-Z]?)")

    ipc_summaries = {}
    for section in ipc_sections:
        match = ipc_df[ipc_df["SectionNumber"] == section]
        if not match.empty:
            desc = match["Description"].values[0]
            summary = summarize_with_local_model(desc)
            ipc_summaries[section] = summary
        else:
            ipc_summaries[section] = "Not found in IPC.csv"

    st.subheader("Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Document Summary")
        st.write(summarized_document)

    with col2:
        st.markdown("### IPC Section Summaries")
        if ipc_summaries:
            for section in sorted(ipc_summaries.keys()):
                with st.expander(f"Section {section}"):
                    st.write(ipc_summaries[section])
        else:
            st.warning("No IPC sections detected in the summary.")
