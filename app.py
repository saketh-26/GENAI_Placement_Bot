# streamlit run app.py
import os
import time
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 1. Page config + configuration loading
# ============================================================
st.set_page_config(page_title="AI Placement Assistant",
                   page_icon="📚",
                   layout="wide")

load_dotenv()

def get_config_value(key, default=None):
    """Read from environment (.env, local) or st.secrets (Streamlit Cloud)."""
    value = os.getenv(key)
    if value:
        return value
    try:
        return st.secrets[key]
    except Exception:
        return default


HF_TOKEN = get_config_value("HF_TOKEN")
MODEL_NAME = get_config_value("MODEL_NAME",
                              "Qwen/Qwen2.5-1.5B-Instruct")

if not HF_TOKEN:
    st.error(
        "HF_TOKEN is missing. Add it to a local `.env` file, or on Streamlit "
        "Community Cloud go to **Settings -> Secrets** and add:\n\n"
        "```\nHF_TOKEN = \"hf_your_token\"\n```"
    )
    st.stop()


# ============================================================
# 2. Create clients/models once when the app starts
# ============================================================
@st.cache_resource(show_spinner=False)
def get_llm_client():
    return InferenceClient(provider="auto", token=HF_TOKEN)


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


client = get_llm_client()
embedding_model = get_embedding_model()


# ============================================================
# 3. Session state
# ============================================================
# Each user's browser session gets its own isolated st.session_state,
# so there's no need for Gradio's manual session_id + global dict + lock.
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "status_message" not in st.session_state:
    st.session_state.status_message = "Upload PDF files to begin."
if "sources_text" not in st.session_state:
    st.session_state.sources_text = ""


# ============================================================
# 4. Build the vector database from uploaded PDF files
# ============================================================
def build_knowledge_base(uploaded_files):
    """
    Load PDFs -> split pages into chunks -> create embeddings ->
    build a FAISS vector database -> create a retriever.
    """
    if not uploaded_files:
        st.session_state.status_message = "Please upload at least one PDF file."
        return

    progress_bar = st.progress(0, text="Loading PDF files")

    try:
        documents = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            for index, uploaded_file in enumerate(uploaded_files, start=1):
                tmp_path = Path(tmp_dir) / uploaded_file.name
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                loader = PyPDFLoader(str(tmp_path))
                loaded_pages = loader.load()

                # Save the readable filename in metadata for source display.
                for page in loaded_pages:
                    page.metadata["source_name"] = uploaded_file.name

                documents.extend(loaded_pages)
                progress_bar.progress(
                    min(0.35, 0.05 + (index / max(len(uploaded_files), 1)) * 0.30),
                    text=f"Loaded {uploaded_file.name}",
                )

            if not documents:
                st.session_state.status_message = "No readable PDF content was found."
                return

            progress_bar.progress(0.45, text="Splitting documents into chunks")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                length_function=len,
            )
            chunks = splitter.split_documents(documents)

            if not chunks:
                st.session_state.status_message = (
                    "The PDFs were loaded, but no text chunks were created."
                )
                return

            progress_bar.progress(0.60, text="Creating embeddings")

            # FAISS.from_documents automatically:
            # 1. creates an embedding for every chunk,
            # 2. builds the FAISS index,
            # 3. links vectors back to their original chunks.
            vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)

            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4},
            )

            st.session_state.retriever = retriever
            st.session_state.chat_history = []
            st.session_state.sources_text = ""

            progress_bar.progress(1.0, text="Knowledge base ready")

            st.session_state.status_message = (
                f"Knowledge base created successfully.\n\n"
                f"PDF files: {len(uploaded_files)}\n"
                f"Pages loaded: {len(documents)}\n"
                f"Chunks created: {len(chunks)}"
            )

    except Exception as error:
        st.session_state.status_message = f"Could not create the knowledge base.\n\nError: {error}"

    finally:
        progress_bar.empty()


# ============================================================
# 5. Generate an answer using the retrieved PDF context
# ============================================================
def generate_answer(question, context, attempts=3):
    """
    Send the question and retrieved context to the hosted LLM.
    Retry temporary API failures before returning an error message.
    """
    system_prompt = (
        "You are an AI Placement Assistant. "
        "Answer only from the supplied document context. "
        "Do not use outside knowledge. "
        "If the answer is not present in the context, say exactly: "
        "\"I don't know based on the uploaded documents.\" "
        "Keep the answer clear and student-friendly."
    )

    user_prompt = f"""
DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=350,
                temperature=0.2,
            )

            answer = response.choices[0].message.content

            if not answer:
                raise ValueError("The model returned an empty response.")

            return answer.strip()

        except Exception as error:
            if attempt < attempts:
                time.sleep(2)
            else:
                return (
                    "The AI service is temporarily unavailable. "
                    f"Please try again.\n\nTechnical details: {error}"
                )


# ============================================================
# 6. Handle each question
# ============================================================
def ask_question(question):
    """
    Retrieve relevant chunks, generate an answer, and update chat history.
    """
    question = (question or "").strip()

    if not question:
        return

    if st.session_state.retriever is None:
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": "Upload PDFs and click 'Build Knowledge Base' first.",
            }
        )
        return

    try:
        relevant_docs = st.session_state.retriever.invoke(question)

        if not relevant_docs:
            answer = "I don't know based on the uploaded documents."
            st.session_state.sources_text = "No relevant source chunks were found."
        else:
            context_parts = []
            source_lines = []

            for number, document in enumerate(relevant_docs, start=1):
                source_name = document.metadata.get(
                    "source_name",
                    Path(document.metadata.get("source", "Unknown PDF")).name,
                )

                # PyPDFLoader stores zero-based page numbers.
                page_number = document.metadata.get("page")
                readable_page = page_number + 1 if isinstance(page_number, int) else "Unknown"

                context_parts.append(
                    f"[Source {number}: {source_name}, page {readable_page}]\n"
                    f"{document.page_content}"
                )
                source_lines.append(f"{number}. {source_name} — page {readable_page}")

            context = "\n\n".join(context_parts)
            answer = generate_answer(question, context)
            st.session_state.sources_text = "\n".join(source_lines)

        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    except Exception as error:
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": f"I could not process that question. Please try again.\n\nError: {error}",
            }
        )
        st.session_state.sources_text = "Retrieval failed."


def clear_session():
    """Remove the vector database and clear the interface."""
    st.session_state.retriever = None
    st.session_state.chat_history = []
    st.session_state.status_message = "Upload PDF files to begin."
    st.session_state.sources_text = ""


# ============================================================
# 7. Build the Streamlit interface
# ============================================================
st.title("📚 AI Placement Assistant with RAG Implementation")
st.caption(
    "Upload one or more placement-related PDFs, create the knowledge base, "
    "and ask questions grounded in those documents."
)

left_col, right_col = st.columns([1, 2])

with left_col:
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    build_clicked = st.button("Build Knowledge Base", type="primary", use_container_width=True)
    clear_clicked = st.button("Clear PDFs and Chat", use_container_width=True)

    if build_clicked:
        build_knowledge_base(uploaded_files)

    if clear_clicked:
        clear_session()
        st.rerun()

    st.text_area(
        "Knowledge Base Status",
        value=st.session_state.status_message,
        height=150,
        disabled=True,
    )

with right_col:
    chat_container = st.container(height=460, border=True)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    question = st.chat_input("Ask a question, e.g. 'Explain only from the uploaded notes.'")
    if question:
        ask_question(question)
        st.rerun()

    st.text_area(
        "Retrieved Sources",
        value=st.session_state.sources_text,
        height=150,
        disabled=True,
    )
