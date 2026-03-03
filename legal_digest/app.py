import dotenv
import requests
import streamlit as st

dotenv.load_dotenv(".env")
import os

API_BACKEND_URL = os.environ.get("API_BACKEND_URL")
st.set_page_config(page_title="Legal Digest", layout="centered")
st.markdown(
    """
<style>
.block-container {
    max-width: 900px;
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
}
.subtitle {
    text-align: center;
    font-size: 16px;
    opacity: 0.8;
    margin-bottom: 30px;
}

div.stButton > button {
    width: 100%;
    height: 70px;               
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.15);
    background: rgba(255,255,255,0.05);
    color: white;
    font-size: 14px;
    padding: 10px;
    
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;

    overflow: hidden;
    text-overflow: ellipsis;
    
    transition: 0.2s ease;
}

div.stButton > button:hover {
    background: #6366f1;
    border: none;
}

.answer-box {
    background: rgba(99,102,241,0.12);
    padding: 1rem;
    border-radius: 12px;
    margin-top: 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("<div class='title'>Legal Digest</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Question Answering on the Constitution of India</div>",
    unsafe_allow_html=True,
)

st.markdown("###### Suggested Questions")

suggested_questions = [
    "What does article 14 of the Constitution of India state?",
    "What is prohibited by article 15 of the Constitution of India?",
    "What is the right to life and personal liberty?",
    "What does article 16 say about equality of opportunity?",
]

col1, col2 = st.columns(2, gap="large")

with col1:
    if st.button(suggested_questions[0], key="q1"):
        st.session_state["selected_question"] = suggested_questions[0]
    if st.button(suggested_questions[2], key="q3"):
        st.session_state["selected_question"] = suggested_questions[2]

with col2:
    if st.button(suggested_questions[1], key="q2"):
        st.session_state["selected_question"] = suggested_questions[1]
    if st.button(suggested_questions[3], key="q4"):
        st.session_state["selected_question"] = suggested_questions[3]

user_query = st.text_area(
    "Enter your question",
    value=st.session_state.get("selected_question", ""),
    height=150,
)

def call_inference_api(query: str):
    try:
        response = requests.post(
            f"{API_BACKEND_URL}/inference/flan-t5-small",
            params={"query": query},
            timeout=60,
        )
        return response.json()
    except Exception as e:
        return {"status": 500, "message": str(e)}

if st.button("Get Answer"):
    if user_query and user_query.strip():
        with st.spinner("Generating answer..."):
            result = call_inference_api(user_query)

        if result["status"] == 200:
            st.markdown(
                f"<div class='answer-box'><b>Answer:</b><br>{result['message']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.error(result["message"])
