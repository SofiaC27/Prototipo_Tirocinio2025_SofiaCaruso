import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from app_functions import *
from ocr_groq import *


load_dotenv("config.env")
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise ValueError("API Key not found!")


# Titolo dell'applicazione
st.markdown("<h1 style='text-align: center; color: blue; font-size: 60px;'>Smart Receipts</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black; font-size: 25px;'>"
            "An advanced web application for uploading receipts and PDFs, extracting data with "
            "OCR, and organizing it in a searchable database. Enhanced with AI/LLM for natural"
            " language interaction and advanced analysis</h2>", unsafe_allow_html=True)


# Upload dei file
st.divider()
st.subheader("File Uploader")

uploaded_files = st.file_uploader("Upload files (JPG, PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
process_uploaded_file(uploaded_files)


# Gestione del database
st.divider()
st.subheader("Database Management")

data = read_data("documents.db", "receipts")
display_data_with_pagination(data)
delete_file_from_database(data)


# OCR
st.divider()
st.subheader("Process files with OCR")

extract_text_from_image(data, api_key)
