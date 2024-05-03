import streamlit as st
from transformers import pipeline
import torch
import re
import pandas as pd
from pypdf import PdfReader

# Load the summarization pipeline
summarizer = pipeline("summarization", model="trained/model", device=torch.device('cpu'), max_length=100)
penal = pd.read_csv("./FIR_DATASET.csv", index_col=0)
penal.dropna(inplace = True)
penal["IPC_Section"] = penal["IPC_Section"].apply(lambda x: x.strip())

articles = pd.read_csv("./articles.csv")
articles["article_id"] = articles["article_id"].apply(lambda x: " ".join(x.split()[:2]))
articles["article_id"] = articles["article_id"].apply(lambda x: x.strip())

# Set the overall width of the app
st.set_page_config(layout="wide")

# Streamlit app
st.title("LegalDigest")

# User input text
input_text = st.text_area("Enter your Legal Document")
input_file = st.file_uploader("Upload a Legal Document")

# Split the input into batches of 512 tokens

if st.button("Summarize"):
    
    if input_text:
        text = input_text
    elif input_file:
        reader = PdfReader(input_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
            
    print(text)
            
    batch_size = 512
    num_batches = (len(text) + batch_size - 1) // batch_size
    summaries = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(text))
        batch_input = text[start_idx:end_idx]

        # Summarize the batch
        res = summarizer(batch_input)

        summaries.append(res[0]['summary_text'])

    # Concatenate the summaries and split into paragraphs
    full_summary = " ".join(summaries)
    full_summary = full_summary.replace("summary:", "\n\n")

    # Store formatted sections in a list
    pattern = r'\b(?:sc|section|Section)\s+\d+\b'
    matches = set(re.findall(pattern, text))

    # Store formatted sections in a set
    formatted_sections = []
    for match in matches:
        if 'sc' in match:
            match = match.replace('sc', 'section')
        sections = re.findall(r'\d{1,3}', match)
        formatted_sections.extend(sections)

    formatted_sections_set = set(map(lambda x: f"IPC Section {x}", formatted_sections))
    
    pattern = r'(article \d+)'
    matches = set(re.findall(pattern, text))

    # Store formatted sections in a set
    formatted_articles = []
    for match in matches:
        ar = re.findall(r'\d{1,3}', match)
        formatted_articles.extend(ar)

    formatted_articles_set = set(map(lambda x: f"Article {x}", formatted_articles))
    
    filtered_articles = articles[articles['article_id'].isin(formatted_articles_set)]
    if not filtered_articles.empty:
        st.markdown("**Important Indian Legal Articles mentioned in the document:**")
        st.dataframe(filtered_articles, use_container_width=True)
    
    filtered_sections = penal[penal['IPC_Section'].isin(formatted_sections_set)]
    if not filtered_sections.empty:
        st.markdown("**Important Indian Legal Sections mentioned in the document:**")
        st.dataframe(filtered_sections, use_container_width=True)
    
    # with col2:
    st.markdown("**Full Summary:**")
    st.write(full_summary)
