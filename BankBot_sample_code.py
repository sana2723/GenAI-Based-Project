import streamlit as st
import pandas as pd
import docx2txt
import PyPDF2
import tempfile
import os
import io

# -------------------------------
# LANGCHAIN IMPORTS
# -------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

# -------------------------------
# CONFIG
# -------------------------------
LLM_MODEL = "gpt-oss:20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RAG_K = 1

# -------------------------------
# STREAMLIT SETUP
# -------------------------------
st.set_page_config(page_title="Banking QA Bot", layout="wide")
st.title("Smart Banking Document QA Bot")
st.caption("Upload PDF, DOCX, TXT, CSV — get answers with source info.")

# -------------------------------
# CACHE MODELS
# -------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource
def load_llm():
    return Ollama(model=LLM_MODEL)

# -------------------------------
# FILE TEXT EXTRACTOR
# -------------------------------
def extract_text_with_meta(file, file_name):
    texts = []
    if file.type == "application/pdf":
        pdf = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                texts.append({
                    "text": text,
                    "page": i,
                    "row": None,
                    "sheet": None,
                    "file": file_name
                })
    elif file.type == "text/plain":
        text = file.getvalue().decode("utf-8")
        texts.append({"text": text, "page": None, "row": None, "sheet": None, "file": file_name})
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            text = docx2txt.process(tmp_path)
            texts.append({"text": text, "page": None, "row": None, "sheet": None, "file": file_name})
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    elif file.type == "text/csv":
        df = pd.read_csv(io.StringIO(file.getvalue().decode("utf-8")))
        for idx, row in df.iterrows():
            texts.append({
                "text": row.to_string(),
                "page": None,
                "row": idx + 1,
                "sheet": None,
                "file": file_name
            })
    return texts

# -------------------------------
# PROMPT TEMPLATE
# -------------------------------
few_shot = """
Example 1:
Q: What is the CECL reserve?
A: $150M in Section 3.1.

Example 2:
Q: What is the LTV?
A: Loan-to-Value ratio must not exceed 80%.
"""

prompt_template = f"""
You are a banking AI assistant.
Answer ONLY using the context below. Include source metadata if applicable.

Rules:
- If answer exists → answer precisely
- If not found → "The requested information is not available."

{few_shot}

Context:
{{context}}

Question:
{{question}}

Answer:
"""

PROMPT = PromptTemplate(input_variables=["context","question"], template=prompt_template)

# -------------------------------
# SESSION STATE
# -------------------------------
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# SIDEBAR UPLOADER
# -------------------------------
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload PDF, DOCX, TXT, CSV", type=["pdf","docx","txt","csv"])
    st.markdown("---")
    if st.session_state.chat_history:
        history_df = pd.DataFrame(st.session_state.chat_history)
        st.download_button(
            "⬇ Export QA History",
            history_df.to_csv(index=False).encode("utf-8"),
            "qa_history.csv",
            "text/csv"
        )

# -------------------------------
# BUILD FAISS + RAG
# -------------------------------
def build_rag_chain(texts, llm):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_chunks = []
    for t in texts:
        chunks = splitter.split_text(t["text"])
        for c in chunks:
            all_chunks.append({**t, "chunk": c})

    embeddings = load_embeddings()
    vectordb = FAISS.from_texts(
        [c["chunk"] for c in all_chunks],
        embeddings,
        metadatas=[{
            "file": c["file"],
            "page": c["page"],
            "row": c["row"],
            "sheet": c["sheet"],
            "source_text": c["chunk"]
        } for c in all_chunks]
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": RAG_K})
    return {"retriever": retriever, "vectordb": vectordb, "llm": llm}

# -------------------------------
# PROCESS UPLOADED FILE
# -------------------------------
if uploaded_file:
    texts = extract_text_with_meta(uploaded_file, uploaded_file.name)
    llm = load_llm()
    st.session_state.rag_chain = build_rag_chain(texts, llm)
    st.success(f"Document indexed: {uploaded_file.name}")

# -------------------------------
# QA INTERFACE
# -------------------------------
if st.session_state.rag_chain:
    st.header("💬 Ask a Question")
    question = st.text_input("Enter your question:")

    if st.button("Submit") and question.strip():
        with st.spinner("Analyzing..."):
            retriever = st.session_state.rag_chain["retriever"]
            llm = st.session_state.rag_chain["llm"]

            # -------------------------------
            # FIXED: 2025 LangChain API
            # -------------------------------
            retrieved_docs = retriever.invoke(question)

            # Combine context for LLM
            context = "\n\n".join([d.page_content for d in retrieved_docs])

            # Use .format() to generate prompt string
            prompt_text = PROMPT.format(context=context, question=question)
            answer = llm.invoke(prompt_text)

            # Save chat history
            # Save chat history
            for d in retrieved_docs:
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "file": d.metadata.get("file"),
                    "page": d.metadata.get("page"),
                    "row": d.metadata.get("row"),
                    "sheet": d.metadata.get("sheet"),
                    "source_text": d.metadata.get("source_text")
                })



# -------------------------------
# DISPLAY CHAT HISTORY AS TABLE
# -------------------------------
if st.session_state.chat_history:
    st.header("📊 Chat History")
    history_df = pd.DataFrame(st.session_state.chat_history)
    st.dataframe(history_df)
