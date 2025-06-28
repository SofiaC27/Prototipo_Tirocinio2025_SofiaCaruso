import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from groq import Groq
import base64
import time
import json
import re
import os

from Database.db_manager import insert_data, get_data


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


def delete_json_from_folder(filename):
    """
    Funzione per eliminare un file JSON specificato dalla cartella 'Extracted_JSON'
    :param filename: nome del file JSON da eliminare
    :return: True se file eliminato, False se non trovato
    """
    file_path = os.path.join(EXTRACTED_JSON_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False


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


def fix_json_data(json_data, base_tolerance=0.10):
    """
    Funzione per controllare, sistemare e validare i dati estratti dal JSON di uno scontrino
    - Calcola il prezzo totale per articoli con quantità > 1 se manca il prezzo complessivo
    - Calcola la percentuale di sconto applicata se non è presente, ma è indicato lo sconto assoluto
    - Calcola lo sconto assoluto se non è presente, ma è indicata la percentuale di sconto
    - Calcola sempre il valore scontato finale in base alla percentuale di sconto o allo sconto assoluto,
      se il valore scontato non è indicato nello scontrino
    - Verifica che la somma dei costi degli articoli sia coerente con il costo totale riportato nello scontrino
    - Mostra un messaggio di avviso se la differenza fra il totale scontrino e la somma dei costi degli articoli
      supera la tolleranza impostata dinamicamente (tiene conto di eventuali errori di lettura degli OCR, adattandosi
      al totale dello scontrino), ma mantiene il valore del totale scontrino originale
    :param json_data: dizionario JSON estratto dallo scontrino
    :param base_tolerance: scarto base accettabile per la differenza fra totale scontrino e somma costi degli articoli
    :return: json_data corretto e validato
    """
    total_items_cost = 0.0

    for item in json_data.get('lista_articoli', []):
        quantity = item.get('quantita') if item.get('quantita') is not None else 1
        price = item.get('prezzo')
        discount_percent = item.get('percentuale_sconto')
        absolute_discount = item.get('sconto_assoluto')
        discounted_value = item.get('valore_scontato')

        # Se quantità > 1 e il prezzo complessivo non è indicato)
        if quantity > 1 and price is not None:
            item['prezzo'] = round(price * quantity, 2)
            price = item['prezzo']  # Aggiorna il prezzo di riferimento

        # Calcola la percentuale di sconto se assente, ma esiste lo sconto assoluto
        if discount_percent is None and absolute_discount is not None:
            if price is not None and price != 0:
                calculated_percent = round((absolute_discount / price) * 100)
                item['percentuale_sconto'] = calculated_percent

        # Calcola lo sconto assoluto se assente, ma esiste la percentuale di sconto
        if absolute_discount is None and discount_percent is not None:
            if price is not None and price != 0:
                calculated_absolute_discount = round(price * discount_percent / 100, 2)
                item['sconto_assoluto'] = calculated_absolute_discount

        # Calcola sempre il valore scontato
        if price is not None:
            if discount_percent is not None:
                discounted_value = round(price - (price * discount_percent / 100), 2)
                item['valore_scontato'] = discounted_value
            elif absolute_discount is not None:
                discounted_value = round(price - absolute_discount, 2)
                item['valore_scontato'] = discounted_value
            else:
                item['valore_scontato'] = None  # Se non ci sono sconti applicati

        # Calcola la somma totale dei prezzi effettivi (usa il prezzo scontato se presente)
        if item.get('valore_scontato') is not None:
            final_price = item['valore_scontato']
        elif item.get('prezzo') is not None:
            final_price = item['prezzo']
        else:
            st.warning(f"Price missing for item {item.get('nome')}. It will be considered 0.")
            final_price = 0.0

        total_items_cost += final_price

    # Confronta con il costo totale riportato nello scontrino
    total_receipt_price = json_data.get('prezzo_totale', {}).get('valore')

    if total_receipt_price is not None:
        dynamic_tolerance = max(base_tolerance, 0.01 * total_receipt_price)

        if abs(total_receipt_price - total_items_cost) > dynamic_tolerance:
            st.warning(
                f"Difference detected between receipt total ({total_receipt_price}) and sum of "
                f"item costs ({round(total_items_cost, 2)}). The original receipt total will be used.")

    return json_data


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


def save_json_to_db(json_data, receipt_id):
    """
    Funzione per salvare i dati estratti dal JSON strutturato nel database
    - Riceve i dati già convertiti in dizionario JSON da un modello AI
    - Inserisce un nuovo record nella tabella 'extracted_data' legandolo a 'receipt_id'
    - Recupera l'ID della riga appena inserita in 'extracted_data'
    - Inserisce ogni prodotto della lista 'lista_articoli' nella tabella 'receipt_items'
      associandolo all'ID della tabella 'extracted_data'
    - Se il record esiste già o c'è un errore, interrompe il flusso e restituisce un messaggio
    :param json_data: dizionario con i dati estratti dal testo OCR strutturato
    :param receipt_id: ID del record esistente nella tabella 'receipts'
    :return: "inserted" se inserimento riuscito, "exists" o "error: ..." in caso di problemi
    """
    extracted_data_row = {
        "receipt_id": receipt_id,
        "purchase_date": json_data.get("data"),
        "purchase_time": json_data.get("ora"),
        "store_name": json_data.get("negozio"),
        "address": json_data.get("indirizzo"),
        "city": json_data.get("città"),
        "country": json_data.get("paese"),
        "total_price": json_data.get("prezzo_totale", {}).get("valore"),
        "total_currency": json_data.get("prezzo_totale", {}).get("valuta"),
        "payment_method": json_data.get("metodo_pagamento")
    }

    result = insert_data("documents.db", "extracted_data", extracted_data_row)
    if result != "inserted":
        return result  # si ferma se il record esiste già o c'è errore

    extracted_data_rows = get_data("documents.db", "extracted_data", ["id"], {"receipt_id": receipt_id})
    extracted_data_id = extracted_data_rows[-1][0] if extracted_data_rows else None
    # [-1][0] per prendere l’ultimo ID appena inserito in extracted_data

    for item in json_data.get("lista_articoli", []):
        item_row = {
            "extracted_data_id": extracted_data_id,
            "name": item.get("nome"),
            "quantity": item.get("quantita"),
            "price": item.get("prezzo"),
            "currency": item.get("valuta"),
            "discount_percent": item.get("percentuale_sconto"),
            "absolute_discount": item.get("sconto_assoluto"),
            "discount_value": item.get("valore_scontato")
        }
        insert_data("documents.db", "receipt_items", item_row)

    return "inserted"


def generate_and_save_json(api_key):
    """
    Funzione per generare, validare e salvare un file JSON a partire dal testo OCR,
    e memorizzare i dati strutturati nel database
    - Verifica che siano presenti il testo estratto e l'immagine selezionata nello session state
    - Mostra un bottone per generare il JSON utilizzando un modello AI tramite l'API Groq
    - Invia il prompt con il testo estratto al modello llama per ottenere i dati strutturati
    - Tenta il parsing del JSON restituito dal modello
    - Se il JSON è valido:
        - Salva il file nella cartella 'Extracted_JSON' evitando duplicati
        - Recupera l'ID dello scontrino dalla tabella 'receipts'
        - Inserisce i dati nella tabella 'extracted_data'
        - Inserisce la lista dei prodotti nella tabella 'receipt_items' (relazionata)
    - Se il JSON non è valido o mancano riferimenti, mostra messaggi di errore
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
            extracted_data_dict = fix_json_data(extracted_data_dict)
            json_content = json.dumps(extracted_data_dict, ensure_ascii=False, indent=2)
            saved_path = save_json_to_folder(json_content, json_filename)
            if saved_path:
                st.success(f"JSON file saved successfully at: {saved_path}")

                rows = get_data("documents.db", "receipts", "Id", {"File_path": st.session_state.selected_image})
                receipt_id = rows[0][0] if rows else None
                # [0][0] per prendere il primo elemento della prima riga, cioè il valore della colonna
                # richiesta (in questo caso "Id")

                if receipt_id is None:
                    st.error("No matching receipt found in database.")
                    return None

                db_result = save_json_to_db(extracted_data_dict, receipt_id)

                if db_result == "inserted":
                    st.success("Data inserted into database.")
                elif db_result == "exists":
                    st.warning("Data already exists in database.")
                else:
                    st.error(f"Database error: {db_result}")

        except json.JSONDecodeError:
            st.error("Generated data is not valid JSON. File not saved.")
            saved_path = None

        return saved_path
