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
    st.session_state.extracted_text = extracted_text

    return extracted_text


def check_data_consistency(json_data_dict):
    """
    Funzione per verificare la coerenza dei dati confrontando il prezzo totale dello scontrino e
    la somma dei prezzi degli articoli
    - Calcola il costo totale degli articoli moltiplicando prezzo e quantità estratti
    - Recupera il prezzo totale salvato dallo scontrino
    - Confronta i due valori e mostra un messaggio di avviso se i dati non sono coerenti e rileva discrepanze
    - Se il prezzo totale è mancante, interrompe il controllo
    :param json_data_dict: dizionario JSON con i dati estratti dallo scontrino
    :return: True se ci sono discrepanze, False se i dati sono coerenti o incompleti
    """
    total_items_cost = 0.0

    for item in json_data_dict.get('lista_articoli', []):
        quantity = item.get('quantità') if item.get('quantità') is not None else 1
        price = item.get('prezzo') if item.get('prezzo') is not None else 0.0
        total_items_cost += price * quantity

    total_receipt_price = json_data_dict.get('prezzo_totale', {}).get('valore')

    # Se manca il prezzo totale, non si può fare il confronto, quindi ritorna False
    if total_receipt_price is None:
        st.warning("Prezzo totale scontrino mancante.")
        return False

    if total_receipt_price != total_items_cost:
        st.warning(
            f"Attenzione: il prezzo totale estratto ({total_receipt_price}) non corrisponde alla somma dei prezzi "
            f"degli articoli ({round(total_items_cost, 2)}).\n"
            "Verifica attentamente i dati, in particolare quantità, prezzi o sconti applicati, e "
            "correggi eventuali errori per garantire accuratezza e coerenza."
        )
        # Se rileva una differenza, ritorna True
        return True

    # Il prezzo totale e la somma dei prezzi degli articoli coincidono (nessuna incoerenza) quindi ritorna False
    return False


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
            "quantity": item.get("quantità"),
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
    Funzione per eseguire l'OCR su uno scontrino e generare il file JSON corrispondente
    - Recupera l'immagine e il percorso dal session state di Streamlit
    - Esegue l'OCR sull'immagine e salva il testo estratto
    - Mostra o nasconde il testo OCR in una textarea attraverso un checkbox
    - Sfrutta il testo estratto per generare un JSON strutturato utilizzando un modello AI tramite l'API Groq
    - Analizza la coerenza dei dati nel JSON
    - Se necessario, mostra un editor Ace JSON per correggere manualmente i dati
    - Salva le modifiche nel session state e visualizza l'anteprima del JSON finale
    :param api_key: chiave per le chiamate API
    """
    image = st.session_state.get("selected_image")
    image_path = st.session_state.get("selected_image_path")

    if not image or not image_path or not os.path.exists(image_path):
        st.warning("Nessuna immagine selezionata o file non trovato.")
        return

    img = Image.open(image_path)

    # === OCR ===
    if st.session_state.ocr_text is None:
        ocr_text = perform_ocr_on_image(api_key)
        if not ocr_text:
            st.error("OCR fallito")
            return
        st.session_state.ocr_text = ocr_text

    # Checkbox per mostrare/nascondere il testo OCR
    show_ocr_checkbox = st.checkbox("Mostra testo OCR estratto", key="show_ocr_text")

    # Se checkbox è attivo, mostra il testo OCR in una textarea
    if st.session_state.show_ocr_text:
        st.text_area("OCR", st.session_state.ocr_text, height=300)

    # === Estrazione JSON ===
    if st.session_state.json_data is None:
        client = Groq(api_key=api_key)
        prompt_text = load_prompt("Modules/AI_prompts/json_prompt.txt")

        # Chiamata al modello AI
        chat_completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "text", "text": st.session_state.ocr_text}
                ]}
            ]
        )

        # Estrazione del contenuto generato
        extracted_data = chat_completion.choices[0].message.content
        raw_json_string = parse_json_from_string(extracted_data.strip())

        try:
            json_data = json.loads(raw_json_string)
            st.session_state.json_data = json_data
            st.session_state.corrected_json_text = json.dumps(json_data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError as e:
            st.error(f"Errore nel parsing del JSON: {e}")
            return

    # === Controlla coerenza dati ===
    needs_correction = check_data_consistency(st.session_state.json_data)

    if needs_correction:
        st.subheader("Verifica e correggi il JSON")

        st.info(
            "**Istruzioni per la modifica:**\n"
            "- Modifica i dati nel formato JSON prestando attenzione alla sintassi: mantieni correttamente"
            " caratteri come virgole, parentesi e virgolette.\n"
            "- Dopo aver effettuato le modifiche, premi prima il bottone 'APPLY' per aggiornare il JSON,"
            " e poi conferma con il bottone 'Salva modifiche'."
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(img, caption=f"Image: {image}", use_container_width=True)

        with col2:
            st.session_state.corrected_json_text = st_ace(
                value=st.session_state.corrected_json_text,
                language="json",
                theme="tomorrow_night",
                height=500,
                key="ace_json_editor"
            )

            if st.button("Salva modifiche"):
                try:
                    corrected_data = json.loads(st.session_state.corrected_json_text)
                    st.session_state.json_data = corrected_data
                    st.success("Modifiche applicate con successo!")
                except json.JSONDecodeError as e:
                    st.error(f"Errore nel JSON modificato: {e}")
    else:
        # Se non servono modifiche, corrected_json_text contiene il JSON formattato
        st.session_state.corrected_json_text = json.dumps(st.session_state.json_data, indent=2, ensure_ascii=False)

    # Visualizza sempre l'anteprima JSON finale (modificato o originale)
    try:
        final_json = json.loads(st.session_state.corrected_json_text)
        st.subheader("Anteprima JSON finale")
        st.json(final_json)
    except json.JSONDecodeError:
        st.error("Errore nel JSON finale, non è possibile visualizzarlo.")

    '''
    # Corregge il JSON e lo salva
    json_filename = os.path.splitext(st.session_state.selected_image)[0] + ".json"
    try:
        extracted_data_dict = json.loads(raw_json_string)
        extracted_data_dict = fix_json_data(extracted_data_dict)
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
    '''


'''
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
    - ...
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

            run_ocr_and_save_json(api_key)

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
'''
