import streamlit as st
import time
import os

from config import IMAGE_DIR, EXTRACTED_JSON_DIR


def init_session_state(defaults):
    """
    Funzione per inizializzare lo stato della sessione Streamlit
    - Controlla se ogni chiave specificata è già presente nello stato della sessione
    - Se una chiave non è presente, la inizializza con il valore di default fornito
    :param defaults: dizionario con coppie chiave-valore da impostare nello stato della sessione
    """
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def show_progress_bar(duration=1.0, message="Elaborazione in corso..."):
    """
    Funzione per mostrare una barra di avanzamento simulata
    - Visualizza un messaggio di caricamento
    - Mostra una barra di avanzamento che si aggiorna gradualmente
    - Simula un'attività di durata controllata
    :param duration: durata totale della barra di avanzamento in secondi
    :param message: messaggio da mostrare durante il caricamento
    """
    with st.spinner(message):
        progress = st.progress(0)
        steps = 100
        for i in range(steps):
            time.sleep(duration / steps)
            progress.progress(i + 1)


def load_prompt_text(file_path):
    """
    Funzione per caricare il file di testo con il prompt da passare all'AI
    - Apre il file in lettura
    - Decodifica in un formato leggibile "utf-8"
    - Rimuove eventuali spazi bianchi o caratteri di nuova riga all'inizio e alla fine del testo
    :param file_path: percorso del file con il prompt da caricare
    :return: stringa di testo corrispondente al prompt
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File prompt non trovato: {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def save_image_to_folder(filename):
    """
    Funzione per salvare delle immagini dentro una cartella apposita
    - Crea la cartella se non esiste già
    - Costruisce il path del file all'interno della cartella
    - Se il file esiste già, non sovrascrive e imposta un flag
    - Salva il file in formato binario per preservarne l’integrità e gestire correttamente
      qualsiasi tipo di dato, inclusi immagini e documenti
    :param filename: file da salvare nella cartella
    :return: percorso del file salvato oppure None se il file esiste già, flag
    """
    os.makedirs(IMAGE_DIR, exist_ok=True)
    file_path = os.path.join(IMAGE_DIR, filename.name)
    if os.path.exists(file_path):
        return None, True
    with open(file_path, "wb") as f:
        f.write(filename.getbuffer())
    return file_path, False


def delete_image_from_folder(filename):
    """
    Funzione per eliminare il file specificato dalla cartella se esiste
    :param filename: nome del file immagine da eliminare
    :return: True se file eliminato, False se non trovato
    """
    file_path = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False


def save_json_to_folder(json_content, filename):
    """
    Funzione per salvare un file JSON in una cartella apposita
    - Crea la cartella se non esiste già
    - Costruisce il percorso completo del file JSON all'interno della cartella
    - Salva il contenuto JSON in formato testo con codifica UTF-8
    - Se il file esiste già, non sovrascrive
    :param json_content: contenuto JSON da salvare (stringa)
    :param filename: nome del file .json
    :return: percorso del file salvato oppure None se il file esiste già
    """
    os.makedirs(EXTRACTED_JSON_DIR, exist_ok=True)
    file_path = os.path.join(EXTRACTED_JSON_DIR, filename)
    if os.path.exists(file_path):
        st.warning(f"Il file JSON '{filename}' esiste già nella cartella. Nessuna azione eseguita.")
        return None
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json_content)
    return file_path


def delete_json_from_folder(filename):
    """
    Funzione per eliminare un file JSON specificato dalla cartella se esiste
    :param filename: nome del file JSON da eliminare
    :return: True se file eliminato, False se non trovato
    """
    file_path = os.path.join(EXTRACTED_JSON_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False
