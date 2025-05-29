import streamlit as st
import pandas as pd
import os
import time

from Database.db_manager import insert_data, delete_data


IMAGE_DIR = "Images"


def save_image_to_folder(uploaded_file):
    """
    Funzione per salvare le immagini inserite con l'upload dentro la cartella 'Images'
    - Crea la cartella 'Images' se non esiste già
    - Costruisce il path del file all'interno della cartella
    - Se il file esiste già, non sovrascrive e imposta un flag
    - Salva il file in formato binario per preservarne l’integrità e gestire correttamente
      qualsiasi tipo di dato, inclusi immagini e documenti
    :param uploaded_file: file caricato da inserire nella cartella
    :return: percorso del file salvato oppure None se il file esiste già, flag
    """
    os.makedirs(IMAGE_DIR, exist_ok=True)
    file_path = os.path.join(IMAGE_DIR, uploaded_file.name)
    if os.path.exists(file_path):
        return None, True
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path, False


def delete_image_from_folder(filename):
    """
    Funzione per eliminare il file specificato dalla cartella se esiste
    :param filename: nome del file da eliminare
    :return: True se file eliminato, False se non trovato
    """
    file_path = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False


def process_uploaded_file(uploaded_files):
    """
    Funzione che gestisce la visualizzazione e il salvataggio dei file caricati
    - Per ogni file da caricare, scrive il nome
    - Se il file da caricare è un'immagine, ne mostra l'anteprima
    - Simula la barra di avanzamento per caricare i file
    - Crea un bottone per salvare i file nel database e nella cartella 'Images'
    - In caso contrario, mostra un warning che invita a fare l'upload per procedere
    - Se il file era già stato caricato e il flag è attivo, mostra un warning e non lo carica nuovamente
    :param uploaded_files: lista di file di cui fare l'upload (può essere anche solo uno)
    """
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write(f"File uploaded: {uploaded_file.name}")

            if uploaded_file.type.startswith("image"):
                st.image(uploaded_file, caption=f"Preview of {uploaded_file.name}", use_container_width=True)

        if st.button("Save all uploaded files"):
            with st.spinner("Saving files..."):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)

                saved_count = 0
                skipped_files_folder = set()
                skipped_files_db = set()

                for uploaded_file in uploaded_files:
                    file_path, already_exists = save_image_to_folder(uploaded_file)
                    if already_exists:
                        skipped_files_folder.add(uploaded_file.name)
                        continue

                    result = insert_data("documents.db", "receipts", {"File_path": uploaded_file.name})
                    if result == "inserted":
                        saved_count += 1
                    elif result == "exists":
                        skipped_files_db.add(uploaded_file.name)

            if saved_count > 0:
                st.success(f"{saved_count} file(s) successfully saved!")

            if saved_count == 0:
                if skipped_files_folder:
                    skipped_list = ", ".join(skipped_files_folder)
                    st.warning(f"The following file(s) already existed in folder and were not saved: {skipped_list}")

                if skipped_files_db:
                    skipped_db_list = ", ".join(skipped_files_db)
                    st.warning(
                        f"The following file(s) already existed in database and were not inserted: {skipped_db_list}")

    else:
        st.warning("Please upload a file to proceed.")


def display_data_with_pagination(data):
    """
    Funzione che mostra i dati salavati nel database con l'impaginazione
    - Recupera i dati presenti nel database (in caso contrario, stampa un messaggio)
    - Crea un dataframe per una visualizzazione migliore dei dati
    - Crea un sistema di impaginazione: la formula calcola il numero totale di pagine necessarie
      per visualizzare tutti gli elementi, arrotondando verso l'alto quando gli elementi non
      sono un multiplo esatto
    - Se il numero totale di pagine è maggiore di 1, mostra i dati in base alla pagina selezionata;
      altrimenti, visualizza tutti i dati senza impaginazione
    :param: data: dati presenti nel database
    """
    st.write("Saved data in the database:")

    if data:
        df = pd.DataFrame(data, columns=["Id", "File_path", "Upload_date"])
        st.dataframe(df)

        items_per_page = 10
        total_pages = (len(data) + items_per_page - 1) // items_per_page
        if total_pages > 1:
            page = st.slider("Page", 1, total_pages)
            start = (page - 1) * items_per_page
            end = start + items_per_page
            for row in data[start:end]:
                st.write(f"Id: {row[0]} | File_path: {row[1]} | Upload_date: {row[2]}")
        else:
            for row in data:
                st.write(f"Id: {row[0]} | File_path: {row[1]} | Upload_date: {row[2]}")
    else:
        st.info("No data available in the database for display.")


def delete_file_from_database_and_folder(data):
    """
    Funzione che permette di selezionare ed eliminare un file dal database
    - Recupera i dati presenti nel database (in caso contrario, stampa un messaggio)
    - Seleziona il file da poter eliminare tra quelli presenti nel database
    - Crea un bottone per eliminare il file
    - Prima della cancellazione chiede conferma, solo in caso affermativo procede a cancellare il file
    :param: data: dati presenti nel database
    """
    if data:
        file_to_delete = st.selectbox("Select file to delete", [row[1] for row in data])

        confirm = st.checkbox(f"Confirm deletion of '{file_to_delete}'")
        st.warning("Please confirm before deleting the file.")

        if confirm:
            if st.button("Delete selected file"):
                delete_data("documents.db", "receipts", "File_path", file_to_delete)
                deleted_from_folder = delete_image_from_folder(file_to_delete)

                st.success(f"File '{file_to_delete}' successfully deleted from database!")
                if deleted_from_folder:
                    st.success(f"File '{file_to_delete}' successfully deleted from the folder!")
    else:
        st.info("No data available in the database for deletion.")
