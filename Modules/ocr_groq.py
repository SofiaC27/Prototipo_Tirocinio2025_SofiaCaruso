import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from groq import Groq
import base64
import time
import json
import re
import os

from Database.db_manager import get_data


IMAGE_DIR = "Images"
EXTRACTED_JSON_DIR = "Extracted_JSON"


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


def save_json_to_folder(json_content, filename):
    """
    Funzione per salvare un file JSON nella cartella 'Extracted_JSON'
    - Crea la cartella 'Extracted_JSON' se non esiste già
    - Costruisce il percorso completo del file JSON all’interno della cartella
    - Salva il contenuto JSON in formato testo con codifica UTF-8
    - Se il file esiste già, non sovrascrive
    :param json_content: contenuto JSON da salvare (stringa)
    :param filename: nome del file .json
    :return: percorso del file salvato oppure None se il file esiste già
    """
    os.makedirs(EXTRACTED_JSON_DIR, exist_ok=True)
    file_path = os.path.join(EXTRACTED_JSON_DIR, filename)
    if os.path.exists(file_path):
        st.warning(f"JSON file '{filename}' already exists in the folder. No action taken.")
        return None
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json_content)
    return file_path


def parse_json_from_string(text):
    """
    Funzione per estrarre il primo oggetto JSON completo dal testo
    - Utilizza regex per cercare un blocco JSON (tra parentesi graffe) nel testo
    :param text: stringa del testo estratto contenente il JSON più eventuale testo extra
    :return: stringa JSON estratta oppure None se non trovato
    """
    pattern = re.compile(r'\{.*\}', re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(0)
    return None


def perform_ocr_on_image(data, api_key):
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
        selected_image = st.selectbox("Select file to process with OCR", [row[1] for row in data])
        file_path = os.path.join(IMAGE_DIR, selected_image)

        img = Image.open(file_path)
        st.image(img, caption=f"Preview of {selected_image}", use_container_width=True)

        if st.button("Run OCR"):
            with st.spinner("Processing OCR..."):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)

            base64_image = encode_image(file_path)
            prompt_text = load_prompt("Modules/AI_prompts/ocr_prompt.txt")

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
            st.session_state.selected_image = selected_image

            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(img, caption=f"Selected Image: {selected_image}", use_container_width=True)
            with col2:
                st.write(f"Extracted text from {selected_image}:")
                st.text(st.session_state.extracted_text)

    else:
        st.info("No data available in the database for processing.")

    return extracted_text, selected_image


def generate_and_save_json(api_key):
    """
    Funzione per generare e salvare un file JSON a partire dal testo estratto con l'OCR
    - Verifica che siano presenti il testo estratto e l'immagine selezionata nello session state
    - Mostra un bottone per generare il JSON utilizzando un modello AI tramite l'API Groq
    - Valida e trasforma il testo estratto in un JSON strutturato
    - Se il JSON è valido, lo salva nella cartella 'Extracted_JSON' evitando duplicati
    - Se il JSON generato non è valido, mostra un messaggio di errore e non salva il file
    :param api_key: chiave per le chiamate API
    :return: percorso del file JSON salvato, oppure None se non salvato
    """
    if not st.session_state.extracted_text or not st.session_state.selected_image:
        st.warning("You must run OCR before generating JSON.")
        return

    client = Groq(api_key=api_key)
    json_filename = os.path.splitext(st.session_state.selected_image)[0] + ".json"

    if st.button(f"Generate JSON for {st.session_state.selected_image}"):
        with st.spinner("Processing JSON..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

        prompt_text = load_prompt("Modules/AI_prompts/json_prompt.txt")

        chat_completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "text", "text": st.session_state.extracted_text}
                ]}
            ]
        )

        extracted_data = chat_completion.choices[0].message.content
        st.session_state.extracted_data = extracted_data

        json_str = parse_json_from_string(extracted_data.strip())
        if not json_str:
            st.error("No JSON object found in extracted data. File not saved.")
            return None

        try:
            extracted_data_dict = json.loads(json_str)
            json_content = json.dumps(extracted_data_dict, ensure_ascii=False, indent=2)
            saved_path = save_json_to_folder(json_content, json_filename)
            if saved_path:
                st.success(f"JSON file saved successfully at: {saved_path}")

                rows = get_data("documents.db", "receipts", "Id", {"File_path": st.session_state.selected_image})
                receipt_id = rows[0][0] if rows else None

                if receipt_id is None:
                    st.error("No matching receipt found in database.")
                    return None

        except json.JSONDecodeError:
            st.error("Generated data is not valid JSON. File not saved.")
            saved_path = None

        return saved_path
