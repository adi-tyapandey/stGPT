import streamlit as st
from streamlit_extras.colored_header import colored_header
from dotenv import load_dotenv
import shutil
import pdfplumber
import PyPDF4
import re
import os
import sys
import tempfile
from typing import Callable, List, Tuple, Dict

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

from google.cloud import storage as gcs_storage

def upload_folder_to_storage(local_folder_path, destination_folder_path):
    # Initialize the Google Cloud Storage client
    gcs_client = gcs_storage.Client.from_service_account_json('serviceAccountKey.json')
    bucket_name = 'pdfgpt-2fdd3.appspot.com'
    bucket = gcs_client.get_bucket(bucket_name)

    for root, dirs, files in os.walk(local_folder_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_file_path = os.path.relpath(local_file_path, local_folder_path)
            destination_file_path = os.path.join(destination_folder_path, relative_file_path)

            print(f"Local file path: {local_file_path}")
            print(f"Destination file path: {destination_file_path}")

            blob = bucket.blob(destination_file_path)
            blob.upload_from_filename(local_file_path)

def extract_metadata_from_pdf(file_path: str) -> dict:
    with open(file_path, "rb") as pdf_file:
        reader = PyPDF4.PdfFileReader(pdf_file)
        metadata = reader.getDocumentInfo()
        return {
            "title": metadata.get("/Title", "").strip(),
            "author": metadata.get("/Author", "").strip(),
            "creation_date": metadata.get("/CreationDate", "").strip(),
        }


def extract_pages_from_pdf(file_path: str) -> List[Tuple[int, str]]:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with pdfplumber.open(file_path) as pdf:
        pages = []
        for page_num, page in enumerate(pdf.pages):
            progress_bar.progress(page_num / (len(pdf.pages)))
            text = page.extract_text()
            if text.strip():  # Check if extracted text is not empty
                pages.append((page_num + 1, text))
    return pages


def parse_pdf(file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    metadata = extract_metadata_from_pdf(file_path)
    pages = extract_pages_from_pdf(file_path)

    return pages, metadata


def merge_hyphenated_words(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def fix_newlines(text: str) -> str:
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def remove_multiple_newlines(text: str) -> str:
    return re.sub(r"\n{2,}", "\n", text)


def clean_text(
    pages: List[Tuple[int, str]], cleaning_functions: List[Callable[[str], str]]
) -> List[Tuple[int, str]]:
    cleaned_pages = []
    for page_num, text in pages:
        for cleaning_function in cleaning_functions:
            text = cleaning_function(text)
        cleaned_pages.append((page_num, text))
    return cleaned_pages


def text_to_docs(text: List[str], metadata: Dict[str, str]) -> List[Document]:
    """Converts list of strings to a list of Documents with metadata."""
    doc_chunks = []

    for page_num, page in text:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(page)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "page_number": page_num,
                    "chunk": i,
                    "source": f"p{page_num}-{i}",
                    **metadata,
                },
            )
            doc_chunks.append(doc)

    return doc_chunks

st.set_page_config(page_title="Upload", page_icon="🗃️")
load_dotenv()
colored_header(
    label="Upload your PDF 🗃️",
    description="",
    color_name="red-70",
    )
st.sidebar.warning('It may take a while when you generate the embeddings for the first time!', icon="⚠️")
col1, col2 = st.columns(2)

with col1:
  pdf_file = st.file_uploader("Interact with your documents with the power of AI", type="pdf")
  if pdf_file is not None:
    file_path = os.path.join(os.getcwd(), pdf_file.name)
    with open(file_path, "wb") as f:
        f.write(pdf_file.read())

    progress_bar = st.progress(0)

    raw_pages, metadata = parse_pdf(file_path)
    cleaning_functions = [merge_hyphenated_words, fix_newlines, remove_multiple_newlines,]
    cleaned_text_pdf = clean_text(raw_pages, cleaning_functions)
    document_chunks = text_to_docs(cleaned_text_pdf, metadata)

    embeddings = HuggingFaceEmbeddings()
    vector_store = Chroma.from_documents(
      document_chunks,
      embeddings,
      collection_name=os.path.splitext(pdf_file.name)[0],
      persist_directory= f"{os.path.join(os.getcwd(), os.path.splitext(pdf_file.name)[0])}/chroma",
    )

    progress_bar.empty()
    vector_store.persist()
    shutil.copy(file_path, f"{os.path.join(os.getcwd(), os.path.splitext(pdf_file.name)[0])}/")
    upload_folder_to_storage(f"{os.path.join(os.getcwd(), os.path.splitext(pdf_file.name)[0])}/", os.path.splitext(pdf_file.name)[0])
    st.balloons()
    st.write(f':green[{pdf_file.name} processed successfully!]')
    st.write(f'Navigate to 📕PDF Chat to ask a Question!')

with col2:
  st.markdown(
      """
      ### How your PDF is processed
      **1) :red[Load data sources to text]:** Loading the PDF and converting it into text.\n
      **2) :red[Chunk text]:** Splitting the loaded text into small chunks.\n
      **3) :red[Embed text]:** Creating a numerical embedding for each chunk of text.\n
      **4) :red[Load embeddings to vectorstore]:** Inserting embeddings and documents into a vectorstore.\n
      """
      )

  st.image('https://blog.langchain.dev/content/images/2023/02/ingest-1.png')
