import streamlit as st
import pandas as pd

from utils import init_session_state, render_sidebar_menu, set_custom_style
from config import DATABASE_NAME

from Database.db_manager import read_data, init_database
from Modules.app_functions import (process_uploaded_file, display_data_with_pagination,
                                   delete_file_from_database_and_folder, display_receipts_data_with_expanders,
                                   show_receipt_predictions)
from Modules.ocr_groq import process_receipt


api_key = st.secrets["general"]["GROQ_API_KEY"]

init_database()

init_session_state({
    "uploaded_files": [],
    "database_data": read_data(DATABASE_NAME, "receipts"),
    "receipts_data": read_data(DATABASE_NAME, "extracted_receipts_data"),
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

set_custom_style()

# Titolo dell'applicazione
st.title("Smart Receipts")
st.markdown("<p style='text-align: center; font-size:20px;'>"
            "Un'applicazione web avanzata per caricare scontrini, estrarre dati tramite OCR e organizzarli in "
            "un database ricercabile. Potenziata con AI/LLM per interazioni in linguaggio naturale "
            "e con ML per analisi avanzate</p>", unsafe_allow_html=True)

render_sidebar_menu()


# Upload dei file
st.divider()
st.header("Caricamento File")

uploaded_files = st.file_uploader("Carica file (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"],
                                  accept_multiple_files=True)

if uploaded_files:
    st.session_state.uploaded_files = uploaded_files  # Aggiorna direttamente la lista
else:
    st.session_state.uploaded_files = []  # Se l'utente rimuove tutto, svuota

process_uploaded_file(st.session_state.uploaded_files)


# Gestione del database
st.divider()
st.header("Gestione del database")

display_data_with_pagination(st.session_state.database_data)


# OCR e JSON
st.divider()
st.header("Elabora i file con OCR e genera JSON")

process_receipt(st.session_state.database_data, api_key)


# Visualizzazione dati degli scontrini
st.divider()
st.header("Visualizzazione dei dati degli scontrini")

display_receipts_data_with_expanders(st.session_state.receipts_data)


# LLM
st.divider()
st.header("Domande in linguaggio naturale con LLM")

st.markdown(
    "Se vuoi avviare l'interfaccia conversazionale con il modello LLM per interrogare il database"
    " [clicca qui per aprire la chat](http://localhost:8000)",
    unsafe_allow_html=True
)


# ML
st.divider()
st.header("Machine Learning")

df = pd.read_csv("Modules/ML/ML_Objects/dataset.csv")
st.dataframe(df)

show_receipt_predictions()


# Eliminazione file
st.divider()
st.header("Gestione file")
delete_file_from_database_and_folder(st.session_state.database_data)
