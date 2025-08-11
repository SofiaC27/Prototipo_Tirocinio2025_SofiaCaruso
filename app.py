import streamlit as st
from PIL import Image
import time
import os

from Database.db_manager import read_data, init_database
from Modules.app_functions import (process_uploaded_file, display_data_with_pagination,
                                   delete_file_from_database_and_folder, display_receipts_data_with_expanders)
from Modules.ocr_groq import run_ocr_and_save_json, ml_predictions_from_json
from Modules.ML.ml_dataset import generate_dataset


IMAGE_DIR = "Images"

init_database()

# Titolo dell'applicazione
st.markdown("<h1 style='text-align: center; color: blue; font-size: 60px;'>Smart Receipts</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black; font-size: 25px;'>"
            "An advanced web application for uploading receipts, extracting data with "
            "OCR, and organizing it in a searchable database. Enhanced with AI/LLM for natural "
            "language interaction and advanced analysis</h2>", unsafe_allow_html=True)


# Upload dei file
st.divider()
st.subheader("File Uploader")


if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

uploaded_files = st.file_uploader("Upload files (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.session_state.uploaded_files = uploaded_files  # Aggiorna direttamente la lista
else:
    st.session_state.uploaded_files = []  # Se l'utente rimuove tutto, svuota

process_uploaded_file(st.session_state.uploaded_files)


# Gestione del database
st.divider()
st.subheader("Database Management")

if "database_data" not in st.session_state:
    st.session_state.database_data = read_data("documents.db", "receipts")

display_data_with_pagination(st.session_state.database_data)


# OCR e JSON
st.divider()
st.subheader("Process files with OCR and generate JSON")

# Inizializzazione stato
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None
if "selected_image_path" not in st.session_state:
    st.session_state.selected_image_path = None
if "start_processing" not in st.session_state:
    st.session_state.start_processing = False
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = None
if "json_data" not in st.session_state:
    st.session_state.json_data = None
if "corrected_json_text" not in st.session_state:
    st.session_state.corrected_json_text = None
if "json_saved" not in st.session_state:
    st.session_state.json_saved = False
if "last_generated_json" not in st.session_state:
    st.session_state.last_generated_json = None
if "trigger_prediction" not in st.session_state:
    st.session_state.trigger_prediction = None


api_key = st.secrets["general"]["GROQ_API_KEY"]

if st.session_state.database_data:

    selected_image = st.selectbox("Select file to process with OCR", [row[1] for row in st.session_state.database_data])
    image_path = os.path.join(IMAGE_DIR, selected_image)

    st.session_state['selected_image'] = selected_image
    st.session_state['selected_image_path'] = image_path

    img = Image.open(image_path)
    st.image(img, caption=f"Preview of {selected_image}", use_container_width=True)

    if st.button(f"OCR + JSON for {selected_image}"):
        with st.spinner("Processing OCR and JSON..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)
        st.session_state.start_processing = True

    if st.session_state.start_processing:
        run_ocr_and_save_json(api_key)

        # Mostra il risultato ML se è stato impostato il trigger
        if st.session_state.get("trigger_prediction", False):
            prediction = ml_predictions_from_json()

            if prediction == 1:
                st.warning(
                    "Questo scontrino è stato classificato come anomalo (outlier). "
                    "Ciò significa che ha caratteristiche insolite rispetto agli altri scontrini. "
                    "Potrebbe indicare un errore nell'OCR, un formato molto diverso o una spesa anomala."
                )
            else:
                st.success(
                    "Questo scontrino è stato classificato come normale. "
                    "Le sue caratteristiche rientrano nella norma rispetto agli altri scontrini."
                )

            # Reset del flag per evitare chiamate ripetute
            st.session_state.trigger_prediction = False

else:
    st.info("No data available in the database for processing.")


# Visualizzazione dati degli scontrini
st.divider()
st.subheader("Displaying Receipt Data")

if "receipts_data" not in st.session_state:
    st.session_state.receipts_data = read_data("documents.db", "extracted_data")
display_receipts_data_with_expanders(st.session_state.receipts_data)


# LLM
st.divider()
st.subheader("Natural language questions with LLM")

st.markdown(
    "If you want to launch the conversational interface with the LLM model to query the database,"
    " [click here to open the chat](http://localhost:8000)",
    unsafe_allow_html=True
)


# ML
st.divider()
st.subheader("Machine Learning")

df = generate_dataset()
st.dataframe(df)


# Eliminazione file
st.divider()
st.subheader("Delete files if needed")
delete_file_from_database_and_folder(st.session_state.database_data)
