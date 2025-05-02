import streamlit as st
import pytesseract
from PIL import Image
import time
import os

IMAGE_DIR = "Images/"  # Cartella delle immagini

# NOTE: configura il percorso dell'eseguibile di Tesseract OCR, necessario per il corretto
# funzionamento della libreria pytesseract per il riconoscimento ottico dei caratteri
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_text_from_image(data):
    """
    Funzione per estrarre il testo da un'immagine attraverso l'OCR
    - Recupera i dati presenti nel database (in caso contrario, stampa un messaggio)
    - Seleziona il file da poter processare tra quelli presenti nel database ed estrae il percorso dell'immagine
    - Crea un bottone per eseguire l'OCR sul file utilizzando la libreria (permette di selezionare la lingua)
    - Crea due colonne per visualizzare l'immagine e il relativo testo
    :param data: dati presenti nel database
    :return: testo estratto dall'immagine
    """
    extracted_text = ""

    if data:
        file_to_process = st.selectbox("Select file to process with OCR", [row[1] for row in data])
        file_path = os.path.join(IMAGE_DIR, file_to_process)

        if st.button("Run OCR"):
            with st.spinner("Processing OCR..."):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)

            img = Image.open(file_path)
            extracted_text = pytesseract.image_to_string(img, lang="eng")

            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(img, caption=f"Selected Image: {file_to_process}", use_container_width=True)
            with col2:
                st.write(f"Extracted text from {file_to_process}:")
                st.text(extracted_text)

    else:
        st.info("No data available in the database for processing.")

    return extracted_text
