import streamlit as st
from PIL import Image
from groq import Groq
import base64
import json
import re
import os
import joblib
import pandas as pd
import mimetypes
from streamlit_ace import st_ace

from utils import show_progress_bar, load_prompt_text, save_json_to_folder
from config import IMAGE_DIR, EXTRACTED_JSON_DIR, DATABASE_NAME

from Database.db_manager import insert_data, get_data
from Modules.ML.ml_dataset import extract_features_from_receipt


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
    prompt_text = load_prompt_text("Modules/AI_prompts/ocr_prompt.txt")

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
      (in caso di sconto, usa il prezzo scontato nel calcolo)
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
        discounted_price = item.get('prezzo_scontato')
        discount_percent = item.get('percentuale_sconto')

        # Se è stato applicato uno sconto, usa il prezzo scontato
        if discounted_price is not None and discount_percent is not None:
            total_items_cost += discounted_price * quantity
        else:
            # Altrimenti usa il prezzo pieno
            total_items_cost += price * quantity

    total_receipt_price = json_data_dict.get('prezzo_totale', {}).get('valore')

    # Se manca il prezzo totale, non si può fare il confronto, quindi ritorna False
    if total_receipt_price is None:
        st.warning("Prezzo totale scontrino mancante")
        return False

    if round(total_receipt_price, 2) != round(total_items_cost, 2):
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

    result = insert_data(DATABASE_NAME, "extracted_receipts_data", extracted_data_row)
    if result != "inserted":
        return result  # si ferma se il record esiste già o c'è errore

    extracted_data_rows = get_data(DATABASE_NAME, "extracted_receipts_data", ["id"], {"receipt_id": receipt_id})
    extracted_data_id = extracted_data_rows[-1][0] if extracted_data_rows else None
    # [-1][0] per prendere l’ultimo ID appena inserito in extracted_data

    for item in json_data.get("lista_articoli", []):
        item_row = {
            "extracted_data_id": extracted_data_id,
            "name": item.get("nome"),
            "quantity": item.get("quantità"),
            "price": item.get("prezzo"),
            "discounted_price": item.get("prezzo_scontato"),
            "currency": item.get("valuta"),
            "discount_percent": item.get("percentuale_sconto")
        }
        insert_data(DATABASE_NAME, "receipt_items", item_row)

    return "inserted"


def manage_json_saving(json_data):
    """
    Funzione per gestire il salvataggio dei dati JSON sia su file che nel database
    - Riceve un dizionario JSON già validato e formattato
    - Salva il contenuto in un file .json nella cartella indicata
    - Recupera l'ID dello scontrino dalla tabella 'receipts' usando il percorso immagine
    - Inserisce i dati nella tabella 'extracted_data' associandoli allo scontrino
    - Aggiorna lo stato della sessione per usi futuri
    - In caso di errore o dati già presenti, mostra un messaggio appropriato
    :param json_data: dizionario con i dati estratti e corretti
    """
    selected_image = st.session_state.get("selected_image")
    json_content = json.dumps(json_data, ensure_ascii=False, indent=2)

    json_filename = os.path.splitext(selected_image)[0] + ".json"
    json_path = save_json_to_folder(json_content, json_filename)

    if json_path:
        st.success(f"File JSON salvato con successo in: {json_path}")

        rows = get_data(DATABASE_NAME, "receipts", "id", {"file_path": selected_image})
        receipt_id = rows[0][0] if rows else None
        # [0][0] per prendere il primo elemento della prima riga, cioè il valore della colonna
        # richiesta (in questo caso "Id")

        if receipt_id is None:
            st.error("Nessuno scontrino trovato nel database")
            return

        db_result = save_json_to_db(json_data, receipt_id)
        if db_result == "inserted":
            st.success("Dati inseriti nel database")
        elif db_result == "exists":
            st.warning("Dati già presenti nel database")
        else:
            st.error(f"Errore database: {db_result}")

        st.session_state.last_generated_json = json_data
        st.session_state.trigger_prediction = True


def extract_json_from_ocr(api_key):
    """
    Funzione per generare un JSON strutturato a partire dal testo OCR
    - Recupera il prompt da file e lo invia al modello AI tramite l'API Groq
    - Riceve una risposta testuale e tenta di estrarre un JSON valido
    - Se il parsing ha successo, salva il JSON nel session state
    - Se il parsing fallisce, mostra un errore e interrompe il flusso
    :param api_key: chiave per le chiamate API
    :return: True se il JSON è stato estratto correttamente, False in caso di errore
    """
    if st.session_state.json_data is None:
        client = Groq(api_key=api_key)
        prompt_text = load_prompt_text("Modules/AI_prompts/json_prompt.txt")

        chat_completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "text", "text": st.session_state.ocr_text}
                ]}
            ]
        )

        extracted_data = chat_completion.choices[0].message.content
        raw_json_string = parse_json_from_string(extracted_data.strip())

        try:
            json_data = json.loads(raw_json_string)
            st.session_state.json_data = json_data
            st.session_state.corrected_json_text = json.dumps(json_data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError as e:
            st.error(f"Errore nel parsing del JSON: {e}")
            return False
    return True


def show_image_and_editor():
    """
    Funzione per modificare manualmente il JSON attraverso un editor
    - Mostra l'immagine in un contenitore scrollabile con zoom personalizzato
    - Accanto all'immagine, presenta un editor Ace JSON per la correzione manuale
    - Dopo la modifica, tenta di validare il JSON e lo salva nel database
    - Evita salvataggi duplicati grazie a un flag nel session state
    """
    image_path = st.session_state.get("selected_image_path")
    img = Image.open(image_path)
    col1, col2 = st.columns([1, 1.3])

    with col1:
        zoom_factor = 0.5
        img_width = int(img.width * zoom_factor)

        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = "image/jpg"

        encoded_img = encode_image(image_path)
        st.markdown(
            f"""
            <div style="height:500px; overflow:auto; border:1px solid #ccc; padding:5px; background-color:white;">
                <img src="data:{mime_type};base64,{encoded_img}" 
                     width="{img_width}px" 
                     style="max-width:none; display:block;">
            </div>
            """,
            unsafe_allow_html=True
        )

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

                # Salva solo se non già salvato
                if not st.session_state.get("json_saved", False):
                    manage_json_saving(corrected_data)
                    st.session_state.json_saved = True  # Marca come salvato

            except json.JSONDecodeError as e:
                st.error(f"Errore nel JSON modificato: {e}")


def run_ocr_and_save_json(api_key):
    """
    Funzione per eseguire l'OCR e gestire il flusso di estrazione e salvataggio dei JSON
    - Verifica la presenza dell'immagine selezionata e ne esegue l'OCR
    - Mostra o nasconde il testo OCR in una textarea attraverso un checkbox
    - Controlla se il file JSON corrispondente all'immagine è già stato salvato
    - Se esiste, lo mostra attraverso un checkbox e ne evita la rigenerazione
    - Se non esiste, genera un JSON strutturato tramite AI e lo valida
    - Se necessario, consente la correzione manuale del JSON
    - Salva i dati finali nella cartella e nel database, evitando duplicazioni
    :param api_key: chiave per le chiamate API
    """
    selected_image = st.session_state.get("selected_image")
    image_path = st.session_state.get("selected_image_path")

    if not selected_image or not image_path or not os.path.exists(image_path):
        st.warning("Nessuna immagine selezionata o file non trovato")
        return

    # Esegue l'OCR
    if st.session_state.ocr_text is None:
        ocr_text = perform_ocr_on_image(api_key)
        if not ocr_text:
            st.error("OCR fallito")
            return
        st.session_state.ocr_text = ocr_text

    # Checkbox per mostrare/nascondere il testo OCR
    if st.checkbox("Mostra testo OCR estratto"):
        st.text_area("OCR", st.session_state.ocr_text, height=300)

    # Controlla se il JSON è già stato salvato
    json_filename = os.path.splitext(selected_image)[0] + ".json"
    json_path = os.path.join(EXTRACTED_JSON_DIR, json_filename)

    if os.path.exists(json_path):
        st.session_state.json_file_exists = True
    else:
        st.session_state.json_file_exists = False

    # Checkbox per mostrare/nascondere il JSON salvato
    if st.session_state.json_file_exists and not st.session_state.json_saved:
        st.warning("Il JSON per questo scontrino è già stato salvato")

        if st.checkbox("Mostra JSON salvato"):
            with open(json_path, "r", encoding="utf-8") as f:
                saved_json = json.load(f)
            st.session_state.json_data = saved_json
            st.session_state.corrected_json_text = json.dumps(saved_json, indent=2, ensure_ascii=False)
            st.json(saved_json)
        return  # Evita di chiamare di nuovo l'API

    # Altrimenti procede all'estrazione del JSON
    if not extract_json_from_ocr(api_key):
        return

    # Controlla coerenza dati e salvataggio JSON
    needs_correction = check_data_consistency(st.session_state.json_data)

    if needs_correction:
        st.subheader("Verifica e correggi il JSON")

        st.info(
            "**Istruzioni per la modifica:**\n"
            "- Usa le barre di scorrimento sull'immagine per visionare tutto lo scontrino e"
            " verificare con facilità i dati presenti.\n"
            "- Modifica i dati nel formato JSON prestando attenzione alla sintassi: mantieni correttamente"
            " caratteri come virgole, parentesi e virgolette.\n"
            "- Dopo aver effettuato le modifiche, premi prima il bottone 'APPLY' per aggiornare il JSON,"
            " e poi conferma con il bottone 'Salva modifiche'."
        )

        show_image_and_editor()

    else:
        # Se non servono modifiche, salva il JSON direttamente (se non è già stato salvato)
        if not st.session_state.json_saved:
            manage_json_saving(st.session_state.json_data)
            st.session_state.json_saved = True  # Marca come salvato


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
    prediction = int(model.predict(X_new_transf)[0])

    return prediction


def process_receipt(data, api_key):
    """
    Funzione per elaborare uno scontrino tramite OCR+JSON e classificazione ML
    - Consente all'utente di selezionare un'immagine da elaborare tra quelle presenti nel database
    - Mostra l'immagine selezionata come anteprima
    - Avvia il processo OCR e genera un file JSON con i dati estratti
    - Se attivato, esegue una classificazione ML per rilevare eventuali anomalie nello scontrino
    - Mostra un messaggio diverso in base al risultato della classificazione
    - Inserisce i dati della previsione nel database
    :param data: dati presenti nel database per la selezione dell'immagine
    :param api_key: chiave per le chiamate API
    """
    if data:
        selected_image = st.selectbox("Seleziona il file da elaborare con OCR", [row[1] for row in st.session_state.database_data])
        image_path = os.path.join(IMAGE_DIR, selected_image)

        st.session_state["selected_image"] = selected_image
        st.session_state["selected_image_path"] = image_path

        img = Image.open(image_path)
        st.image(img, caption=f"Anteprima di {selected_image}", use_container_width=True)

        if st.button(f"OCR + JSON per {selected_image}"):
            show_progress_bar(duration=1.5, message="Elaborazione OCR e JSON in corso...")
            st.session_state.start_processing = True

        if st.session_state.start_processing:
            run_ocr_and_save_json(api_key)

            '''
            # Mostra il risultato ML se è stato impostato il trigger
            if st.session_state.get("trigger_prediction", False):
                st.subheader("Previsione Machine Learning")
                prediction = ml_predictions_from_json()

                if prediction == 1:
                    st.markdown(
                        "<span style='color:red; font-weight:bold;'>Scontrino anomalo (outlier)</span><br>"
                        "Questo scontrino presenta caratteristiche insolite rispetto agli altri. "
                        "Può indicare un errore nell'OCR, un formato molto diverso o una spesa anomala",
                        unsafe_allow_html=True
                    )
                    prediction_label = "Anomalo (outlier)"
                else:
                    st.markdown(
                        "<span style='color:green; font-weight:bold;'>Scontrino normale</span><br>"
                        "Le caratteristiche di questo scontrino rientrano nella norma rispetto agli altri",
                        unsafe_allow_html=True
                    )
                    prediction_label = "Normale"

                # Salvataggio nel database
                insert_data("documents.db", "receipt_predictions", {
                    "file_name": st.session_state.selected_image,
                    "prediction": prediction,
                    "prediction_label": prediction_label
                })

                # Reset del flag per evitare chiamate ripetute
                st.session_state.trigger_prediction = False
            '''

    else:
        st.info("Nessun dato disponibile nel database per l'elaborazione")
