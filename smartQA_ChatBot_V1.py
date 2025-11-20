import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceEndpoint
from langchain import PromptTemplate
import os
import yaml
import box
from dotenv import find_dotenv, load_dotenv
import timeit

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open(r'D:\GenAI_Model\N_LLM\config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

# Function to initialize session state
def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

# Initialize session state
init_session_state()

# Build the LLM
def build_llm():
    HUGGINGFACEHUB_API_TOKEN = ' '
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"

    llm = HuggingFaceEndpoint(repo_id=model_id, max_new_tokens=128, temperature=0.7, token=HUGGINGFACEHUB_API_TOKEN)
    return llm

# Set QA prompt
def set_qa_prompt():
    qa_template = """Use the following pieces of information to answer the user's question.
    Do not provide answers more than 100 words.
    Try to be specific and accurate.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Helpful answer:
    """
    prompt = PromptTemplate(template=qa_template, input_variables=['context', 'question'])
    return prompt

# Build retrieval QA
def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT}),
                                       return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
                                       chain_type_kwargs={'prompt': prompt})
    return dbqa

# Setup DBQA
def setup_dbqa():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = build_llm()
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)
    return dbqa

# Add to chat history
def add_to_chat_history(chat_history, query, response, source_docs):
    for doc in source_docs:
        chat_history.append({
            'Question': query,
            'Answer': response["result"],
            'Document Name': doc.metadata["source"],
            'Page Number': doc.metadata["page"] + 1,
            'Source Text': doc.page_content
        })

# Main application
def main():
    st.markdown("<h1 style='text-align: center;'>🤖Questions Answer Bot</h1>", unsafe_allow_html=True)

    # Sidebar for uploading files and settings
    with st.sidebar:
        st.header("Upload PDF Files")
        
        # Upload PDF files
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.session_state.uploaded_files.append(uploaded_file)
                
                pdf_reader = PdfReader(uploaded_file)
                num_pages = len(pdf_reader.pages)
                st.write(f"Number of pages in {uploaded_file.name}: {num_pages}")

                # Progress bar for reading pages
                progress_bar = st.progress(0)
                
                for i, page in enumerate(pdf_reader.pages):
                    # Simulate some processing time
                    progress_bar.progress((i + 1) / num_pages)
                    
                st.success(f"Finished processing {uploaded_file.name}")

    # Main area for user interaction
    #st.write("### Ask a question about the uploaded documents:")
    query = st.text_input('Enter your question here:')
    
    if st.button('Get Answer'):
        with st.spinner('Processing your question...'):
            dbqa = setup_dbqa()
            response = dbqa({'query': query})
        
        st.markdown("**Answer:**")
        st.write(response["result"])
        st.markdown("---")

        # Process source documents
        source_docs = response['source_documents']
        add_to_chat_history(st.session_state.chat_history, query, response, source_docs)

        # Display chat history along with current question-answer pair
        st.markdown("### Chat History")
        df_chat_history = pd.DataFrame(st.session_state.chat_history)
        
        st.dataframe(df_chat_history)

        # Allow users to download the chat history as CSV
        csv_chat_history = df_chat_history.to_csv(index=False)
        st.download_button(
            label="Download Chat History CSV",
            data=csv_chat_history,
            file_name='chat_history.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()
