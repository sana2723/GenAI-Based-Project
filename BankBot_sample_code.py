import streamlit as st
import os
import io
from tempfile import NamedTemporaryFile
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub # For remote LLM access

# --- Configuration & Secrets ---
# Replace 'YOUR_HF_TOKEN' with st.secrets if deploying to Streamlit Cloud
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN_PLACEHOLDER"))

# Mistral-7B Instruct Model ID from Hugging Face Hub (fast and high quality)
MISTRAL_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# --- Helper Functions ---

# Mapping file extensions to LangChain Loaders
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".xlsx": UnstructuredExcelLoader,
}

@st.cache_resource
def load_and_process_documents(uploaded_files):
    """Loads, splits, and embeds documents into a FAISS vector store."""
    if not uploaded_files:
        st.warning("Please upload documents to begin.")
        return None

    all_texts = []
    
    # Process uploaded files
    for uploaded_file in uploaded_files:
        try:
            # Use a temporary file to allow LangChain loaders to access the file path
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext not in LOADER_MAP:
                st.warning(f"Skipping unsupported file type: {uploaded_file.name}")
                continue
            
            with NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            loader = LOADER_MAP[ext](tmp_file_path)
            documents = loader.load()
            all_texts.extend(documents)
            
            os.remove(tmp_file_path) # Clean up temp file
            
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
            return None

    if not all_texts:
        return None
        
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(all_texts)

    # Create Hugging Face Embeddings
    st.info("Creating embeddings using Hugging Face's 'all-MiniLM-L6-v2'...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create FAISS Vector Store
    vectorstore = FAISS.from_documents(docs, embeddings)
    st.success(f"Successfully processed {len(all_texts)} pages/records and created FAISS vector store.")
    return vectorstore

@st.cache_resource
def get_qa_chain(vectorstore):
    """Initializes the RetrievalQA chain with Mistral-7B."""
    if not vectorstore:
        return None
        
    st.info(f"Connecting to Mistral-7B ({MISTRAL_MODEL_ID}) via HuggingFace Hub...")
    
    try:
        # Initialize the LLM using HuggingFaceHub (requires API Token)
        llm = HuggingFaceHub(
            repo_id=MISTRAL_MODEL_ID,
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            model_kwargs={"temperature": 0.1, "max_length": 512}
        )
    except Exception as e:
        st.error(f"LLM Connection Error: {e}. Check your HUGGINGFACEHUB_API_TOKEN.")
        return None

    # Create the Retrieval QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), # Retrieve top 3 documents
        return_source_documents=True
    )
    return qa_chain

# --- Streamlit Application ---

def main():
    st.set_page_config(page_title="🏦 GenAI Smart QA Bot", layout="wide")

    st.title("🤖 GenAI Smart QA Bot: Banking Document Analysis")
    st.markdown("---")

    # --- Sidebar for File Upload and Configuration ---
    with st.sidebar:
        st.header("📂 Document Loader")
        uploaded_files = st.file_uploader(
            "Upload banking documents:",
            type=["pdf", "txt", "csv", "docx", "xlsx"],
            accept_multiple_files=True
        )
        st.markdown("---")
        st.caption(f"**LLM:** Mistral-7B (HuggingFace Hub)")
        st.caption(f"**Vector DB:** FAISS")
        st.caption(f"**Framework:** LangChain, Streamlit")

    # Initialize vector store and QA chain
    vectorstore = load_and_process_documents(uploaded_files)
    qa_chain = get_qa_chain(vectorstore)

    # --- Main Chat Interface ---
    if qa_chain:
        
        # 1. Initialize Session State for Chat History and CSV Export
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "export_data" not in st.session_state:
            st.session_state.export_data = []

        # Display Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("source_info"):
                     st.caption(message["source_info"])

        # 2. Handle User Input
        if prompt := st.chat_input("Ask a domain-specific query (e.g., 'What is the required credit score for commercial loans?')..."):
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Process the query
            with st.spinner("Analyzing documents with Mistral-7B..."):
                try:
                    result = qa_chain.invoke({"query": prompt})
                    response = result['result']
                    
                    # Construct detailed source information
                    source_info = "🔍 **Sources Found:**\n"
                    for i, doc in enumerate(result['source_documents']):
                         file_source = os.path.basename(doc.metadata.get('source', 'Unknown File'))
                         page_info = f"Page {doc.metadata.get('page', 'N/A')}" if 'page' in doc.metadata else ""
                         
                         source_info += f"* **{file_source}** ({page_info}): *...{doc.page_content[:150].replace('\n', ' ')}...*\n"
                         
                except Exception as e:
                    response = f"An error occurred: {e}"
                    source_info = "Could not retrieve sources due to error."

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
                st.caption(source_info)
            st.session_state.messages.append({"role": "assistant", "content": response, "source_info": source_info})
            
            # 3. Update CSV Export Data
            st.session_state.export_data.append({
                "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Query": prompt,
                "Answer": response,
                "Sources": '; '.join([os.path.basename(doc.metadata.get('source', 'N/A')) for doc in result.get('source_documents', [])]) if 'result' in locals() else "N/A"
            })


        # --- CSV Export Button (Using a placeholder for the "Real-time model training" concept) ---
        if st.session_state.export_data:
            st.markdown("---")
            st.info("💡 **Model Adaptability:** The chat log can be used as a dataset for real-time model training/fine-tuning (e.g., via human feedback loops on the QA pairs).")
            
            df_export = pd.DataFrame(st.session_state.export_data)
            csv_data = df_export.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="⬇️ Export Chat Log (for CSV Export Feature)",
                data=csv_data,
                file_name='smart_qa_bot_log.csv',
                mime='text/csv',
                help="Download the history of all QA pairs for analysis or future fine-tuning data."
            )
            
    else:
        st.info("Please upload your banking documents in the sidebar and ensure your Hugging Face API token is set to initialize the LLM.")

if __name__ == "__main__":
    main()