import streamlit as st

from app_functions import *
from ocr_groq import *


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
delete_file_from_database(st.session_state.database_data)


# OCR
st.divider()
st.subheader("Process files with OCR")

api_key = st.secrets["general"]["GROQ_API_KEY"]

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

extracted_text, selected_image = extract_text_from_image(st.session_state.database_data, api_key)

st.session_state.extracted_text = extracted_text
st.session_state.selected_image = selected_image

if st.session_state.extracted_text:
    extract_data_to_json(api_key)
