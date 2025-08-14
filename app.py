import streamlit as st

from utils import init_session_state

from Database.db_manager import read_data, init_database
from Modules.app_functions import (process_uploaded_file, display_data_with_pagination,
                                   delete_file_from_database_and_folder, display_receipts_data_with_expanders,
                                   show_receipt_predictions)
from Modules.ocr_groq import process_receipt
from Modules.ML.ml_dataset import generate_dataset


api_key = st.secrets["general"]["GROQ_API_KEY"]

init_database()

init_session_state({
    "uploaded_files": [],
    "database_data": read_data("documents.db", "receipts"),
    "receipts_data": read_data("documents.db", "extracted_data"),
    "selected_image": None,
    "selected_image_path": None,
    "start_processing": False,
    "ocr_text": None,
    "json_data": None,
    "corrected_json_text": None,
    "json_saved": False,
    "last_generated_json": None,
    "trigger_prediction": None,
    "json_file_exists": False
})


# Titolo dell'applicazione
st.markdown("<h1 style='text-align: center; color: blue; font-size: 60px;'>Smart Receipts</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black; font-size: 25px;'>"
            "Un'applicazione web avanzata per caricare scontrini, estrarre dati tramite OCR e organizzarli in "
            "un database ricercabile. Potenziata con AI/LLM per interazioni in linguaggio naturale "
            "e analisi avanzate</h2>", unsafe_allow_html=True)


# Upload dei file
st.divider()
st.subheader("Caricamento File")

uploaded_files = st.file_uploader("Upload files (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.session_state.uploaded_files = uploaded_files  # Aggiorna direttamente la lista
else:
    st.session_state.uploaded_files = []  # Se l'utente rimuove tutto, svuota

process_uploaded_file(st.session_state.uploaded_files)


# Gestione del database
st.divider()
st.subheader("Gestione del Database")

display_data_with_pagination(st.session_state.database_data)


# OCR e JSON
st.divider()
st.subheader("Elabora i file con OCR e genera JSON")

process_receipt(st.session_state.database_data, api_key)


# Visualizzazione dati degli scontrini
st.divider()
st.subheader("Visualizzazione dei dati degli scontrini")

display_receipts_data_with_expanders(st.session_state.receipts_data)


# LLM
st.divider()
st.subheader("Domande in linguaggio naturale con LLM")

st.markdown(
    "Se vuoi avviare l'interfaccia conversazionale con il modello LLM per interrogare il database"
    " [clicca qui per aprire la chat](http://localhost:8000)",
    unsafe_allow_html=True
)


# ML
st.divider()
st.subheader("Machine Learning")

df = generate_dataset()
st.dataframe(df)

show_receipt_predictions()


# Eliminazione file
st.divider()
st.subheader("Elimina i file se necessario")
delete_file_from_database_and_folder(st.session_state.database_data)
