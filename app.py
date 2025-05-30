import streamlit as st

from Database.db_manager import read_data, init_database
from Modules.app_functions import (process_uploaded_file, display_data_with_pagination,
                                   delete_file_from_database_and_folder)
from Modules.ocr_groq import perform_ocr_on_image, generate_and_save_json


init_database()

# Titolo dell'applicazione
st.markdown("<h1 style='text-align: center; color: blue; font-size: 60px;'>Smart Receipts</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black; font-size: 25px;'>"
            "An advanced web application for uploading receipts and PDFs, extracting data with "
            "OCR, and organizing it in a searchable database. Enhanced with AI/LLM for natural"
            " language interaction and advanced analysis</h2>", unsafe_allow_html=True)


# Upload dei file
st.divider()
st.subheader("File Uploader")


if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

uploaded_files = st.file_uploader("Upload files (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.session_state.uploaded_files.extend(uploaded_files)

process_uploaded_file(st.session_state.uploaded_files)


# Gestione del database
st.divider()
st.subheader("Database Management")

if "database_data" not in st.session_state:
    st.session_state.database_data = read_data("documents.db", "receipts")

display_data_with_pagination(st.session_state.database_data)


# OCR
st.divider()
st.subheader("Process files with OCR")

api_key = st.secrets["general"]["GROQ_API_KEY"]

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = ""

extracted_text, selected_image = perform_ocr_on_image(st.session_state.database_data, api_key)

if extracted_text:
    st.session_state.extracted_text = extracted_text
if selected_image:
    st.session_state.selected_image = selected_image

generate_and_save_json(api_key)


# Eliminazione file
st.divider()
st.subheader("Delete files if needed")
delete_file_from_database_and_folder(st.session_state.database_data)
