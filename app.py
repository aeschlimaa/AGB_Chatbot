import os
import streamlit as st
from agb_chatbot import get_rag_components

# Initialize the RAG pipeline
chain, retriever, pages, documents = get_rag_components()

# Title of the app
st.title("RAG Document Assistant")
st.write(
    "This app allows you to query the contents of uploaded PDFs using a Retrieval-Augmented Generation (RAG) pipeline powered by OpenAI or Ollama models."
)

# Info about loaded docs
st.success(f"✅ Loaded {len(documents)} pages from PDF files.")
st.info(f"🧠 Chunked into {len(pages)} text segments ready for retrieval.")

# Input for the question
question = st.text_input("💬 What is your question?", placeholder="e.g. What are the cancellation terms?")

# Query the RAG pipeline
if question:
    with st.spinner("Thinking..."):
        try:
            response = chain.invoke({"question": question})
            st.session_state.response = response
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Display the response
if "response" in st.session_state and st.session_state.response:
    st.subheader("📄 Response")
    st.write(st.session_state.response)

