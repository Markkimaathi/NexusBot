import os
import streamlit as st
import pickle
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="NexusBot: Research Tool 📈", page_icon="🤖")
st.title("NexusBot: Chat with URLs and PDFs 📄🔗")
st.sidebar.title("Input Options")

# OpenAI API Key setup
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(temperature=0.7, max_tokens=500)

# URLs Input
st.sidebar.subheader("Enter 3 URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    if url.strip():
        urls.append(url)

# PDFs Input
st.sidebar.subheader("Upload PDF files")
pdf_docs = st.sidebar.file_uploader("Upload your PDFs", accept_multiple_files=True)

# Buttons
process_inputs_clicked = st.sidebar.button("Process Inputs")

# File path to save FAISS index
file_path = "faiss_store_openai.pkl"

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to process URLs and PDFs and build vector store
def process_inputs(urls, pdf_docs):
    all_text = ""

    # Processing URLs
    if urls:
        loader = UnstructuredURLLoader(urls=urls)
        url_data = loader.load()
        url_text = " ".join([doc.page_content for doc in url_data])
        all_text += url_text

    # Processing PDFs
    if pdf_docs:
        pdf_text = get_pdf_text(pdf_docs)
        all_text += pdf_text

    if all_text:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_text(all_text)

        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(docs, embeddings)

        # Save vector store
        with open(file_path, "wb") as f:
            pickle.dump(vector_store, f)
        st.success("Processing complete! You can now ask questions.")
    else:
        st.warning("Please enter at least one URL or upload a PDF.")

# Process inputs when button is clicked
if process_inputs_clicked:
    process_inputs(urls, pdf_docs)

# Query the processed data
query = st.text_input("Ask a question about the processed data:")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
        retriever = vectorstore.as_retriever()
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

        # Run the query and display results
        result = chain({"question": query}, return_only_outputs=True)

        st.subheader("Answer")
        st.write(result.get("answer", "No answer found."))

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
        else:
            st.write("No sources available.")
    else:
        st.warning("Please process the URLs and PDFs first.")
