from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from google.cloud import storage as gcs_storage

import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_toggle import st_toggle_switch
from gpt4free import usesless
import shutil
import pdfplumber
import PyPDF4
import re
import os
import sys
import tempfile
from typing import Callable, List, Tuple, Dict
from dotenv import load_dotenv
import base64
from pdf2image import convert_from_path

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional

class CustomLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        req = usesless.Completion.create(prompt=prompt, parentMessageId="")
        try:
            if stop is not None:
                for stop_word in stop:
                    if stop_word in req['text']:
                        generated_text = req['text'].split(stop_word)[0]
                        return generated_text
                # If none of the stop words are found, return the original text
                return req['text']
            else:
                return req['text']
        except KeyError:
            # Retry the request when KeyError occurs
            return self._call(prompt, stop, run_manager)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

def download_folder_from_storage(source_folder_path, local_folder_path):
    gcs_client = gcs_storage.Client.from_service_account_json('serviceAccountKey.json')
    bucket_name = 'pdfgpt-2fdd3.appspot.com'
    bucket = gcs_client.get_bucket(bucket_name)

    if not source_folder_path.endswith('/'):
        source_folder_path += '/'

    blobs = bucket.list_blobs(prefix=source_folder_path)

    for blob in blobs:
        relative_file_path = blob.name[len(source_folder_path):]
        destination_file_path = os.path.join(local_folder_path, relative_file_path)

        if os.path.exists(destination_file_path):
            continue

        os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
        blob.download_to_filename(destination_file_path)

def list_folders_in_storage():
    gcs_client = gcs_storage.Client.from_service_account_json('serviceAccountKey.json')
    bucket_name = 'pdfgpt-2fdd3.appspot.com'
    bucket = gcs_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs()
    folder_names = set()
    for blob in blobs:
        folder_path = os.path.dirname(blob.name)
        if '/' in folder_path:
            folder_name = folder_path.split('/')[0]
            folder_names.add(folder_name)
    return folder_names

def make_chain(selected_folder):
    global vector_store
    model = CustomLLM()
    embedding = HuggingFaceEmbeddings()

    vector_store = Chroma(
        collection_name=selected_folder,
        embedding_function=embedding,
        persist_directory= f"{os.getcwd()}/{selected_folder}/chroma",
    )

    return ConversationalRetrievalChain.from_llm(
        model,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        # verbose=True,
    )

def start_chain(selected_folder):
  download_folder_from_storage(selected_folder, f'{os.getcwd()}/{selected_folder}')
  chain = make_chain(selected_folder)
  return chain

def display_pdf(pdf_path, page_number):
  images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
  if images:
    st.image(images[0], caption=f"Page {page_number}", use_column_width=True)
  else:
    st.error("Failed to convert the specified page.")

st.set_page_config(page_title="Chat", page_icon="📕")
load_dotenv()
if "chat_history" not in st.session_state:
  st.session_state["chat_history"] = []
colored_header(
    label="Interactive PDF Chat 🔍📚",
    description="Explore your documents using the sidebar",
    color_name="red-70",
)
folders = list_folders_in_storage()
selected_folder = st.sidebar.selectbox("Select a Document 📕", list(folders))
if selected_folder:
  st.session_state["chat_history"] = []
  chain = start_chain(selected_folder)
  with st.form("question_form"):
    user_question = st.text_area(f"Ask a question about the {selected_folder}:")
    submit_button = st.form_submit_button(label="Submit", type="secondary", use_container_width=True)
  if user_question and submit_button:
    with st.spinner('Loading...'):
      response = chain({"question": user_question, "chat_history": st.session_state["chat_history"]})
      answer = response["answer"]
      source = response["source_documents"]
      st.session_state["chat_history"].append(HumanMessage(content=user_question))
      st.session_state["chat_history"].append(AIMessage(content=answer))
    st.write(answer)
    with st.expander("Sources:"):
      for i in range(len(source)):
        page_content = source[i].page_content
        page_number = source[i].metadata['page_number']
        col1, col2 = st.columns(2)
        with col1:
          st.write(f"**:red[Page {page_number}]:** {page_content}")
        with col2:
          pdf_path = f"{os.getcwd()}/{selected_folder}/{selected_folder}.pdf"
          display_pdf(pdf_path, page_number)
