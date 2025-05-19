import os
import base64
from dotenv import load_dotenv
from groq import Groq
import streamlit as st
from PIL import Image
import time


IMAGE_DIR = "Images/"


def encode_image(img_path):
    """
    Funzione per codificare l'immagine in Base64
    - Apre il file in lettura binaria
    - Legge il contenuto e lo converte in una stringa in base 64
    - Decodifica in un formato leggibile "utf-8"
    :param img_path: percorso dell'immagine da codificare
    :return: stringa in base 64 dell'immagine
    """
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_prompt(file_path):
    """
    Funzione per caricare il file di testo con il prompt da passare all'AI
    - Apre il file in lettura
    - Decodifica in un formato leggibile "utf-8"
    - Rimuove eventuali spazi bianchi o caratteri di nuova riga all'inizio e alla fine del testo
    :param file_path: percorso del file con il prompt da caricare
    :return: stringa di testo corrispondente al prompt
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def extract_text_from_image(data, api_key):
    """
    Funzione per estrarre il testo da un'immagine attraverso l'OCR
    - Recupera l'API Key dall'ambiente e controlla se è presente
    - Inizializza il client Groq
    - Recupera i dati presenti nel database (in caso contrario, stampa un messaggio)
    - Seleziona il file da poter processare tra quelli presenti nel database ed estrae il percorso dell'immagine
    - Crea un bottone per eseguire l'OCR sul file utilizzando Groq e llama4
    - Crea due colonne per visualizzare l'immagine e il relativo testo sottoforma
    :param data: dati presenti nel database
    :param api_key: chiave per le chiamate API
    :return: testo estratto dall'immagine
    """
    client = Groq(api_key=api_key)

    extracted_text = ""

    if data:
        file_to_process = st.selectbox("Select file to process with OCR", [row[1] for row in data])
        file_path = os.path.join(IMAGE_DIR, file_to_process)

        img = Image.open(file_path)
        st.image(img, caption=f"Preview of {file_to_process}", use_container_width=True)

        if st.button("Run OCR"):
            with st.spinner("Processing OCR..."):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)

            base64_image = encode_image(file_path)
            prompt_text = load_prompt("prompt.txt")

            chat_completion = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ]
            )

            extracted_text = chat_completion.choices[0].message.content

            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(img, caption=f"Selected Image: {file_to_process}", use_container_width=True)
            with col2:
                st.write(f"Extracted text from {file_to_process}:")
                st.text(extracted_text)

    else:
        st.info("No data available in the database for processing.")

    return extracted_text
