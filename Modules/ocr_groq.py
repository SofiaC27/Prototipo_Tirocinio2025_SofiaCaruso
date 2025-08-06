import streamlit as st
from PIL import Image
from groq import Groq
import base64
import time
import json
import re
import os
import joblib
import pandas as pd
from streamlit_ace import st_ace

from Database.db_manager import insert_data, get_data
from Modules.ML.ml_dataset import extract_features_from_receipt


IMAGE_DIR = "Images"
EXTRACTED_JSON_DIR = "Extracted_JSON"


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


def perform_ocr_on_image(api_key):
    """
    Funzione per estrarre il testo da un'immagine attraverso l'OCR
    - Recupera il percorso dell'immagine selezionata dallo stato della sessione Streamlit
    - Codifica l'immagine in base64 per l'invio al modello AI tramite il client Groq
    - Esegue l'OCR ed estrae il testo
    :param api_key: chiave per le chiamate API
    :return: testo estratto tramite OCR
    """
    image_path = st.session_state.get("selected_image_path")

    client = Groq(api_key=api_key)
    base64_image = encode_image(image_path)
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

    return extracted_text


def fix_json_data(api_key, json_data_dict, ocr_text):
    """
    Funzione per verificare e correggere la coerenza tra testo OCR e dati JSON estratti
    - Recupera l'immagine e i dati estratti dallo stato della sessione Streamlit
    - Converte il JSON in una stringa formattata
    - Invia il testo OCR e iol JSON al modello Groq per fare validazione semantica
    - Se i dati sono coerenti, conferma il contenuto
    - Se ci sono discrepanze, mostra l'immagine e il JSON per una modifica manuale
    - Consente la correzione diretta tramite editor Ace e conferma finale
    :param api_key: chiave per le chiamate API
    :param json_data_dict: dizionario estratto contenente i dati dello scontrino
    :param ocr_text: testo estratto tramite OCR
    :return: dizionario JSON finale validato e corretto
    """
    image = st.session_state.get("selected_image")
    image_path = st.session_state.get("selected_image_path")
    img = Image.open(image_path)

    client = Groq(api_key=api_key)
    prompt_text = load_prompt("Modules/AI_prompts/comparison_prompt.txt")
    json_string = json.dumps(json_data_dict, indent=2, ensure_ascii=False)

    chat_completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "text", "text": f"TESTO OCR:\n{ocr_text}"},
                {"type": "text", "text": f"JSON ESTRATTO:\n{json_string}"}
            ]}
        ]
    )

    comparison = chat_completion.choices[0].message.content.strip()
    final_json_dict = json_data_dict

    # Se i dati sono coerenti, ritorna il dizionario originale
    if "DATI COERENTI" in comparison.upper():
        st.success("I dati estratti sono coerenti con il testo OCR")
    # Altrimenti, permette all'utente di correggere il JSON
    else:
        st.warning("Sono state rilevate differenze tra il testo OCR e i dati estratti")
        st.info("Controlla e correggi i dati nel campo qui sotto prima di salvarli")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption=f"Image: {image}", use_container_width=True)
        with col2:
            st.write("Dati JSON estratti (modificabili):")

            if "corrected_json_text" not in st.session_state:
                st.session_state.corrected_json_text = json_string

            st.session_state.corrected_json_text = st_ace(
                value=st.session_state.corrected_json_text,
                language="json",
                theme="tomorrow_night",
                height=400,
                key="ace_json_fix"
            )

        # Conferma modifica
        if st.button("Conferma dati corretti"):
            try:
                st.session_state.corrected_json_final = json.loads(st.session_state.corrected_json_text)
                st.success("Dati aggiornati correttamente")
            except Exception as e:
                st.error(f"Errore nel JSON modificato: {e}")
                st.stop()

        # Blocca finché non è stato confermato qualcosa
        if "corrected_json_final" not in st.session_state:
            st.info("Conferma i dati per continuare")
            st.stop()

        final_json_dict = st.session_state.corrected_json_final

    return final_json_dict


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


def run_ocr_and_save_json(api_key):
    """
    Funzione che esegue l'OCR su uno scontrino e genera il file JSON corrispondente
    - Recupera l'iimmagine e il percorso dallo stato della sessione Streamlit
    - Applica l'OCR usando Groq e un'immagine codificata in base64
    - Mostra il testo OCR estratto, se richiesto, e valida che non sia vuoto
    - Genera un JSON strutturato a partire dal testo OCR
    - Salva il file JSON nella cartella
    - Inserisce i dati estratti nel database associandoli allo scontrino originale
    :param api_key: chiave per le chiamate API
    :return: dizionario con dati strutturati, oppure None in caso di errore
    """
    image = st.session_state.get("selected_image")
    image_path = st.session_state.get("selected_image_path")

    if not image or not image_path or not os.path.exists(image_path):
        st.warning("Nessuna immagine selezionata o file non trovato.")
        return

    # Esegue l'OCR
    ocr_text = perform_ocr_on_image(api_key)

    # Mostra opzionalmente il testo OCR
    if st.checkbox(f"Mostra testo OCR estratto da {image}"):
        st.text_area("Testo OCR", ocr_text, height=200)

    if not ocr_text.strip():
        st.error("Testo OCR vuoto. Impossibile continuare.")
        return

    # Chiamata al modello Groq per generare JSON
    json_filename = os.path.splitext(st.session_state.selected_image)[0] + ".json"
    client = Groq(api_key=api_key)
    prompt_text = load_prompt("Modules/AI_prompts/json_prompt.txt")

    chat_completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "text", "text": ocr_text}
            ]}
        ]
    )

    extracted_data = chat_completion.choices[0].message.content
    raw_json_string = parse_json_from_string(extracted_data.strip())

    if not raw_json_string:
        st.error("No JSON object found in extracted data. File not saved.")
        return None

    # Corregge il JSON e lo salva
    try:
        extracted_data_dict = json.loads(raw_json_string)
        extracted_data_dict = fix_json_data(api_key, extracted_data_dict, ocr_text)
        json_content = json.dumps(extracted_data_dict, ensure_ascii=False, indent=2)
        json_path = save_json_to_folder(json_content, json_filename)
        if json_path:
            st.success(f"JSON file saved successfully at: {json_path}")

            rows = get_data("documents.db", "receipts", "Id", {"File_path": image})
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

            st.session_state.last_generated_json = extracted_data_dict
            st.session_state.trigger_prediction = True

    except json.JSONDecodeError:
        st.error("Generated data is not valid JSON. File not saved.")
        extracted_data_dict = None

    return extracted_data_dict


def ml_predictions_from_json():
    """
    Funzione per effettuare la predizione su uno scontrino a partire da un file JSON:
    - Carica scaler, encoder e modello ML salvati in locale
    - Estrae le feature rilevanti dallo scontrino
    - Codifica le variabili categoriche con OneHotEncoder
    - Costruisce il vettore delle feature nell'ordine atteso dal modello
    - Trasforma le feature con lo scaler per normalizzarle
    - Esegue la predizione con il modello (0 = normale, 1 = anomalo)
    :return: risultato della previsione come valore intero, oppure None in caso di errore
    """
    if "last_generated_json" not in st.session_state or not st.session_state.last_generated_json:
        return None

    json_data = st.session_state.last_generated_json

    # Carica scaler, modello e encoder
    scaler = joblib.load("Modules/ML/ML_Objects/scaler.joblib")
    model = joblib.load("Modules/ML/ML_Objects/final_model.joblib")
    encoder = joblib.load("Modules/ML/ML_Objects/encoder.joblib")

    # Estrae le feature come dizionario
    feature_dict = extract_features_from_receipt(json_data)
    if feature_dict is None:
        return None

    df = pd.DataFrame([feature_dict])

    # Codifica la colonna categorica usando l'encoder salvato
    encoded = encoder.transform(df[['season']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['season']), index=df.index)
    df = pd.concat([df.drop(columns=['season']), encoded_df], axis=1)

    X_new = df.drop(['date'], axis=1).values

    # Trasforma le feature e fa la previsione
    X_new_transf = scaler.transform(X_new)
    prediction = model.predict(X_new_transf)[0]

    return prediction


def process_receipt(data, api_key):
    """
    Funzione per gestire l'interfaccia utente e il flusso OCR/JSON
    - Mostra le immagini selezionabili da elaborare
    - Visualizza l’immagine corrente
    - Consente di eseguire l’OCR e generare il JSON con pulsante dedicato
    - Mostra una barra di caricamento durante l’elaborazione
    - Esegue la classificazione ML se il flag è attivo
    - Mostra messaggio finale in base alla predizione
    :param data: dati presenti nel database
    :param api_key: chiave per le chiamate API
    """
    if data:
        selected_image = st.selectbox("Select file to process with OCR", [row[1] for row in data])
        image_path = os.path.join(IMAGE_DIR, selected_image)
        st.session_state['selected_image'] = selected_image
        st.session_state['selected_image_path'] = image_path

        img = Image.open(image_path)
        st.image(img, caption=f"Preview of {selected_image}", use_container_width=True)

        if st.button(f"OCR and JSON for {selected_image}"):
            with st.spinner("Processing OCR and JSON..."):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)

                extracted_data_dict = run_ocr_and_save_json(api_key)
                st.session_state["last_generated_json"] = extracted_data_dict

        if st.session_state.get("trigger_prediction", False):
            prediction = ml_predictions_from_json()

            if prediction == 1:
                st.warning("Questo scontrino è stato classificato come anomalo (outlier). "
                           "Ciò significa che ha caratteristiche insolite rispetto agli altri scontrini. "
                           "Potrebbe indicare un errore nell'OCR, un formato molto diverso o una spesa anomala.")
            else:
                st.success("Questo scontrino è stato classificato come normale. "
                           "Le sue caratteristiche rientrano nella norma rispetto agli altri scontrini.")

            # Reset del flag per evitare chiamate ripetute
            st.session_state.trigger_prediction = False

    else:
        st.info("No data available in the database for processing.")
