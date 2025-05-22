import os
import base64
from dotenv import load_dotenv
from groq import Groq
import streamlit as st
from PIL import Image
import time
import json


IMAGE_DIR = "Images/"
EXTRACTED_JSON_DIR = "Extracted_JSON/"


def get_api_key():
    """
    Funzione per caricare l'API Key
    - Carica il file con le variabili d'ambiente
    - Estrae il valore dell'API Key e controlla che esista, altrimenti dà errore
    :return: API Key
    """
    load_dotenv("config.env")
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError("API Key not found!")

    return api_key


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
    - Crea due colonne per visualizzare l'immagine e il relativo testo
    :param data: dati presenti nel database
    :param api_key: chiave per le chiamate API
    :return: testo estratto dall'immagine
    :return: immagine selezionata
    """
    client = Groq(api_key=api_key)

    extracted_text = ""
    selected_image = None

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
            prompt_text = load_prompt("App/AI_prompts/ocr_prompt.txt")

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
            st.session_state.extracted_text = extracted_text
            st.session_state.selected_image = file_to_process

            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(img, caption=f"Selected Image: {file_to_process}", use_container_width=True)
            with col2:
                st.write(f"Extracted text from {file_to_process}:")
                st.text(st.session_state.extracted_text)

    else:
        st.info("No data available in the database for processing.")

    return extracted_text, selected_image


def extract_data_to_json(api_key, extracted_text, selected_image):
    """
    Funzione per convertire il testo estratto in formato JSON
    - Crea il nome del file JSON basato sul nome dell'immagine corrispondente e genera il
      percorso completo in cui verrà salvato, nella cartella Extracted_JSON
    - Crea un bottone per visualizzare il JSON corrispondente al testo estratto dall'immaigne selezionata
    :param api_key: chiave per le chiamate API
    :param extracted_text: testo estratto
    :param selected_image: immagine selezionata
    :return: dati estratti dal testo in formato JSON
    """
    client = Groq(api_key=api_key)

    # json_filename = os.path.splitext(image_name)[0] + ".json"
    # json_path = os.path.join(EXTRACTED_JSON_DIR, json_filename)

    with st.spinner("Processing JSON..."):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

    prompt_text = load_prompt("App/AI_prompts/json_prompt.txt")

    chat_completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "text", "text": extracted_text}
            ]}
        ]
    )

    extracted_data = chat_completion.choices[0].message.content

    st.success(f"JSON generated for {selected_image}")
    st.text(extracted_data)

    return extracted_data
