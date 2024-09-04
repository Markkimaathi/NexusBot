import os
import streamlit as st
import pickle
import time
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

st.title("NexusBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Collect up to 3 URLs from the user
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():  # Check if the URL is not empty
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    if urls:  # Ensure there are valid URLs
        try:
            # Load data from the URLs
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.text("Data Loading... Started... ✅✅✅")
            data = loader.load()

            # Split the data into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Text Splitting... Started... ✅✅✅")
            docs = text_splitter.split_documents(data)

            # Create embeddings and save them to a FAISS index
            embeddings = OpenAIEmbeddings()
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            main_placeholder.text("Embedding Vectors... Started Building... ✅✅✅")
            time.sleep(2)

            # Save the FAISS index to a pickle file
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_openai, f)

            main_placeholder.success("Processing complete. You can now ask questions.")
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
    else:
        st.warning("Please enter at least one valid URL.")

# Allow the user to input a query
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        try:
            # Load the FAISS index from the pickle file
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)

                # Display the answer and sources
                st.header("Answer")
                st.write(result.get("answer", "No answer found."))

                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")  # Split sources by newline
                    for source in sources_list:
                        st.write(source)
                else:
                    st.write("No sources available.")
        except Exception as e:
            st.error(f"An error occurred while retrieving the answer: {e}")
    else:
        st.warning("The FAISS index does not exist. Please process the URLs first.")
