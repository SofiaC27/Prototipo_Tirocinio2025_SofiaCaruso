import streamlit as st
import pandas as pd
import os

from utils import show_progress_bar, save_image_to_folder, delete_image_from_folder, delete_json_from_folder
from Database.db_manager import insert_data, delete_data, get_data


def paginate(data, page_key, items_per_page=10):
    """
    Funzione per gestire l'impaginazione dei dati
    - Inizializza la pagina corrente nello stato della sessione se non è già presente
    - Calcola il numero totale di pagine in base alla lunghezza dei dati e agli elementi per pagina
    - Mostra i pulsanti "Previous" e "Next" per navigare tra le pagine
    - Visualizza il numero della pagina corrente
    :param data: lista di elementi da impaginare
    :param page_key: chiave univoca per tracciare la pagina corrente nello stato della sessione
    :param items_per_page: numero di elementi da mostrare per pagina (default: 10)
    :return: solo gli elementi relativi alla pagina selezionata
    """
    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    total_pages = (len(data) + items_per_page - 1) // items_per_page
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("Previous", key=f"prev_{page_key}", use_container_width=True) and st.session_state[page_key] > 1:
            st.session_state[page_key] -= 1

    with col3:
        if st.button("Next", key=f"next_{page_key}", use_container_width=True) and st.session_state[page_key] < total_pages:
            st.session_state[page_key] += 1

    st.write(f"Page {st.session_state[page_key]} of {total_pages}")

    start = (st.session_state[page_key] - 1) * items_per_page
    end = start + items_per_page
    return data[start:end]


def display_image_gallery(images, columns=5):
    """
    Funzione che gestisce la visualizzazione di una galleria di immagini con selezione
    - Se non sono presenti immagini mostra un messagio informativo
    - Sincronizza le immagini caricate nello stato della sessione per mantenere la preview
    - Mostra una galleria di immagini organizzate in griglia dinamica
    - Ogni immagine è visualizzata con la propria didascalia (caption) ed è automaticamente cliccabile
      tramite la funzionalità nativa di Streamlit (icona fullscreen in alto a destra sull'immagine)
    - Aggiunge un messaggio informativo per spiegare all'utente che può ingrandire le immagini tramite l'icona nativa
    :param images: lista di immagini caricate
    :param columns: numero di colonne della griglia (impostato a 5)
    """
    if not images:
        st.info("No images to display.")
        return

    # Sincronizza la galleria con la sessione
    st.session_state.uploaded_files_for_preview = images

    st.subheader("Image Gallery")
    st.info("Click on the fullscreen icon (top-right of each image) to enlarge and preview the image.")

    rows = (len(images) + columns - 1) // columns

    for row_idx in range(rows):
        cols = st.columns(columns)
        for col_idx in range(columns):
            img_idx = row_idx * columns + col_idx
            if img_idx < len(images):
                file = images[img_idx]
                with cols[col_idx]:
                    st.image(file, caption=file.name, use_container_width=True)


def process_uploaded_file(uploaded_files):
    """
    Funzione che gestisce la visualizzazione e il salvataggio dei file caricati
    - Se sono presenti file da caricare, li salva nello stato della sessione per mostrarne la preview
    - Mostra un'anteprima dinamica (tipo galleria) delle immagini caricate, ma non ancora salvate
    - Imposta un flag per evitare che la preview venga ripetuta dopo il salvataggio
    - Crea un bottone per salvare i file nel database e nella cartella apposita
    - Durante il salvataggio, visualizza una barra di avanzamento
    - Se un file è già presente nella cartella o nel database, non lo salva di nuovo e mostra un avviso
    - Se invece non sono presenti file da caricare, mostra un warning che invita a fare l'upload per procedere
    :param uploaded_files: lista di file di cui fare l'upload (può essere anche solo uno)
    """
    if uploaded_files:
        # Salva i file caricati solo per la preview
        if "uploaded_files_for_preview" not in st.session_state:
            st.session_state.uploaded_files_for_preview = []

        if "files_saved" not in st.session_state:
            st.session_state.files_saved = False

        # Aggiorna solo se ci sono nuovi file caricati
        if uploaded_files != st.session_state.uploaded_files_for_preview:
            st.session_state.uploaded_files_for_preview = uploaded_files
            st.session_state.files_saved = False  # Resetta il flag per consentire una nuova preview

        # Mostra le preview solo se i file non sono ancora stati salvati
        if not st.session_state.files_saved:
            st.write("Preview (not saved yet):")
            display_image_gallery(st.session_state.uploaded_files_for_preview)

            if st.button("Save all uploaded files"):
                st.session_state.files_saved = True  # Dopo il salvataggio, blocca le preview

                show_progress_bar(duration=1.5, message="Saving files...")

                saved_count = 0
                skipped_files_folder = set()
                skipped_files_db = set()

                for uploaded_file in st.session_state.uploaded_files_for_preview:
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
            st.info("Files already saved. Upload new files to preview again.")

    else:
        st.warning("Please upload a file to proceed.")


def display_data_with_pagination(data):
    """
    Funzione che mostra i dati salavati nel database con l'impaginazione
    - Recupera i dati presenti nel database (in caso contrario, stampa un messaggio)
    - Crea un dataframe per una visualizzazione migliore dei dati
    - Usa un sistema di impaginazione
    - Se il numero totale di pagine è maggiore di 1, mostra i dati in base alla pagina selezionata;
      altrimenti, visualizza tutti i dati senza impaginazione
    :param: data: dati presenti nel database
    """
    st.write("Saved data in the database:")

    if data:
        df = pd.DataFrame(data, columns=["Id", "File_path", "Upload_date"])
        st.dataframe(df)

        for row in paginate(data, page_key="uploads"):
            st.write(f"Id: {row[0]} | File_path: {row[1]} | Upload_date: {row[2]}")

    else:
        st.info("No data available in the database for display.")


def display_receipts_data_with_expanders(receipts_data):
    """
    Funzione che mostra i dati degli scontrini con visualizzazione espandibile e impaginazione
    - Verifica la presenza di dati (in caso contrario, mostra un messaggio informativo)
    - Usa un sistema di impaginazione
    - Estrae solo gli scontrini appartenenti alla pagina corrente
    - Per ogni scontrino, crea una sezione espandibile con i dettagli principali
    - Recupera e visualizza gli articoli associati allo scontrino tramite una tabella
      all'interno dell'espander
    :param receipts_data: elenco di tuple contenenti le informazioni degli scontrini da visualizzare
    """
    if receipts_data:
        receipts_to_display = paginate(receipts_data, page_key="receipts")

        for receipt in receipts_to_display:
            receipt_id = receipt[0]
            receipt_number = receipt[1]
            purchase_date = receipt[2]
            purchase_time = receipt[3]
            store_name = receipt[4]
            address = receipt[5]
            city = receipt[6]
            country = receipt[7]
            total_price = receipt[8]
            total_currency = receipt[9]
            payment_method = receipt[10]

            with st.expander(f"Receipt {receipt_number}"):
                st.write(f"**Purchase date:** {purchase_date}")
                st.write(f"**Purchase time:** {purchase_time}")
                st.write(f"**Store:** {store_name}")
                st.write(f"**Address:** {address}, {city}, {country}")
                st.write(f"**Total:** {total_price} {total_currency}")
                st.write(f"**Payment method:** {payment_method}")

                # Recupera gli articoli collegati allo scontrino
                receipt_items = get_data(
                    db_name="documents.db",
                    table_name="receipt_items",
                    columns=[
                        "name", "quantity", "price", "currency",
                        "discount_percent", "absolute_discount", "discount_value"
                    ],
                    conditions={"extracted_data_id": receipt_id}
                )

                if receipt_items:
                    df_items = pd.DataFrame(receipt_items, columns=[
                        "name", "quantity", "price", "currency",
                        "discount_percent", "absolute_discount", "discount_value"
                    ])
                    st.dataframe(df_items, use_container_width=True)

    else:
        st.info("No receipt data saved in the database.")


def delete_file_from_database_and_folder(data):
    """
    Funzione che permette di selezionare ed eliminare un file dal database
    - Recupera i dati presenti nel database (in caso contrario, stampa un messaggio)
    - Seleziona il file immagine da poter eliminare tra quelli presenti nel database
    - Crea un bottone per eliminare il file immagine
    - Prima della cancellazione chiede conferma, solo in caso affermativo procede a cancellare il file immagine
    :param: data: dati presenti nel database
    """
    if data:
        file_to_delete = st.selectbox("Select file to delete", [row[1] for row in data])

        confirm = st.checkbox(f"Confirm deletion of '{file_to_delete}'")
        st.warning("Please confirm before deleting the file.")

        if confirm:
            if st.button("Delete selected file"):
                delete_data("documents.db", "receipts", {"File_path": file_to_delete})
                st.success(f"File '{file_to_delete}' successfully deleted from database!")

                deleted_from_folder = delete_image_from_folder(file_to_delete)
                if deleted_from_folder:
                    st.success(f"Image '{file_to_delete}' successfully deleted from the folder!")

                possible_json = os.path.splitext(file_to_delete)[0] + ".json"
                deleted_json = delete_json_from_folder(possible_json)
                if deleted_json:
                    st.success(f"Associated JSON file '{possible_json}' successfully deleted from the folder!")

    else:
        st.info("No data available in the database for deletion.")
