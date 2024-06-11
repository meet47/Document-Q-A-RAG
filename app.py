#Import libraries
import streamlit as st
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.document_loaders import PyPDFLoader
import tempfile
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

#Load environment variables
load_dotenv()

#Groq Client
groq_api_key=os.environ['GROQ_API_KEY']

#Function to read content of PDF File
def read_pdf(file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    
    os.remove(temp_file_path)
    
    return documents

#Streamlit App
st.title("Meet's Document Q&A RAG App")

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

pdf = st.file_uploader("Upload a PDF file", type="pdf")

if pdf is not None and st.session_state.pdf_text is None:
    st.session_state.pdf_text = read_pdf(pdf)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    splitted_text = text_splitter.split_documents(st.session_state.pdf_text)
    st.session_state.vector_store = FAISS.from_documents(splitted_text, HuggingFaceBgeEmbeddings())

user_input = st.text_input("Ask you question based on uploaded PDF")

if user_input and st.session_state.vector_store is not None:

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context. 
    Think step by step before providing a detailed answer. 
    I will tip you $1000 if the user finds the answer helpful. 
    Answer only if you find the answer. Otherwise say I couldn't find the answer.
    <context>
    {context}
    </context>
    Question: {input}
                                        """)

    # Chat Model Llama3-8b
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-8b-8192"
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": user_input})

    # Output for the User
    st.write(response['answer'])


