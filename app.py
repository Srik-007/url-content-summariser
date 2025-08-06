import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import os
from dotenv import load_dotenv
import re

def remove_think_block(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# Load .env variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Safety check
if not api_key:
    st.error("GROQ_API_KEY not found. Please check your .env file.")
    st.stop()

# Initialize LLM (use correct parameter names)
llm = ChatGroq(
    model_name="deepseek-r1-distill-llama-70b",  # Correct parameter name is model_name, not model
    temperature=0.7,
    api_key=api_key
)

# Prompt setup
prompt_template = """
Provide an informative summary of the given content in 300 words:
Content:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Streamlit UI
st.set_page_config(page_title="URL SUMMARISER", page_icon="🌕", layout="wide")
st.title("URL SUMMARISER")
st.subheader("Summarise any YouTube or website URL")

url = st.text_input("Paste the URL here", label_visibility="visible")

if st.button("Summarise"):
    if not url.strip():
        st.error("No URL entered.")
    elif not validators.url(url):
        st.error("Invalid URL format.")
    else:
        try:
            with st.spinner("Fetching and summarising content..."):
                # Loader selection
                if "youtube.com" in url or "youtu.be" in url:
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0","User-Agent":"Chrome/116.0.5845.188 Safari/537.36"}
                    )

                data = loader.load()
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(data)
                output_summary=remove_think_block(output_summary)

                st.success("Summary:")
                st.write(output_summary)

        except Exception as e:
            st.error("An error occurred:")
            st.exception(e)