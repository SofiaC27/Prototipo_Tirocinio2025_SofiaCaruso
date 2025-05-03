import streamlit as st
import pytesseract
from PIL import Image
import time
import os
import cv2
import numpy as np
from spellchecker import SpellChecker

IMAGE_DIR = "Images/"  # Cartella delle immagini

# NOTE: configura il percorso dell'eseguibile di Tesseract OCR, necessario per il corretto
# funzionamento della libreria pytesseract per il riconoscimento ottico dei caratteri
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def analyze_image(file_path):
    """
    Funzione per analizzare un'immagine a cui applicare il pre-processing
    - Converte l'immagine in scala di grigi
    - Ricava l'istogramma dell'immagine
    - Applica la treshold binaria per convertire l'immagine in bianco e nero
    - Verifica il valore del contrasto e, se basso, applica l'equalizzazione
    :param file_path: percorso dell'immagine da pre-processare
    :return: immagine pre-processata
    """
    #img = cv2.imread(file_path)
    img = Image.open(file_path)
    img_np = np.array(img)

    gray_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

    #binary_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)[1]
    binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    contrast_level = np.std(hist)
    if contrast_level < 30:
        binary_img = cv2.equalizeHist(binary_img)

    return binary_img


def correct_text(text):
    """
    Funzione per correggere il testo estratto tramite OCR utilizzando la libreria pyspellchecker
    - Controlla il testo (permette di selezionare la lingua => tra cui it, en)
    - Divide il testo, ogni volta che incontra uno spazio, in una lista di parole
    - Corregge, se c'è bisogno, le parole e le aggiugne alla lista delle parole corrette
    - Ricostruisce il testo corregendo gli errori
    :param text: testo estratto dall'immagine
    :return: testo corretto
    """
    spell = SpellChecker(language='it')
    words = text.split()
    corrected_words = []

    for word in words:
        if word:
            corrected_word = spell.correction(word)
            if corrected_word is None:
                corrected_word = word
            corrected_words.append(corrected_word)
        else:
            corrected_words.append(word)

    corrected_text = " ".join(corrected_words)

    return corrected_text


def extract_text_from_image(data):
    """
    Funzione per estrarre il testo da un'immagine attraverso l'OCR
    - Recupera i dati presenti nel database (in caso contrario, stampa un messaggio)
    - Seleziona il file da poter processare tra quelli presenti nel database ed estrae il percorso dell'immagine
    - Permette di selezionare se applicare il pre-processing oppure no
    - Crea un bottone per eseguire l'OCR sul file utilizzando la libreria pytesseract
      (permette di selezionare la lingua => tra cui ita, eng)
    - Crea due colonne per visualizzare l'immagine e il relativo testo
    :param data: dati presenti nel database
    :return: testo estratto dall'immagine
    """
    #extracted_text = ""
    corrected_text = ""

    if data:
        file_to_process = st.selectbox("Select file to process with OCR", [row[1] for row in data])
        file_path = os.path.join(IMAGE_DIR, file_to_process)

        img_original = Image.open(file_path)
        st.image(img_original, caption=f"Preview of {file_to_process}", use_container_width=True)
        apply_preprocessing = st.checkbox("Apply Pre-processing")

        if st.button("Run OCR"):
            with st.spinner("Processing OCR..."):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)

            if apply_preprocessing:
                img = analyze_image(file_path)
            else:
                img = img_original

            config = r'--oem 3 --psm 4'
            extracted_text = pytesseract.image_to_string(img, config=config, lang="eng")
            corrected_text = correct_text(extracted_text)

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.image(img, caption=f"Selected Image: {file_to_process}", use_container_width=True)
            with col2:
                st.write(f"Extracted text (raw OCR) from {file_to_process}:")
                st.text(extracted_text)
            with col3:
                st.write(f"Corrected text from {file_to_process}:")
                st.text(corrected_text)

    else:
        st.info("No data available in the database for processing.")

    return corrected_text #extracted_text
