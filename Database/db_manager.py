import sqlite3


def create_table(db_name, table_query):
    """
    Funzione per creare una tabella
    - Connessione al database
    - Creazione di un cursore per eseguire le query
    - Query per creare la tabella se non esiste già
    - Salvataggio dei cambiamenti e chiusura della connessione
    :param: db_name: nome del database
    :param: table_query: query per creare una tabella
    """
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute(table_query)
    conn.commit()
    conn.close()


def get_connection(db_name):
    """
    Funzione per connettersi al database
    :param: db_name: nome del database
    :return: connessione al database
    """
    return sqlite3.connect(db_name)  # Return a connection to the database


def insert_data(db_name, table_name, data_dict):
    """
    Funzione per inserire dati all'interno del database
    - Connessione al database
    - Creazione di un cursore per eseguire le query
    - Costruzione dinamica della query SQL INSERT in base alle chiavi e ai valori forniti
    - Utilizzo di segnaposto (?) per prevenire SQL injection
    - Inserimento dei dati nella tabella specificata
    - Se è presente un vincolo di unicità e l'inserimento fallisce, viene gestito l'errore come duplicato
    - Salvataggio dei cambiamenti e chiusura della connessione
    :param db_name: nome del database
    :param table_name: nome della tabella dove inserire i dati
    :param data_dict: dizionario con i dati da inserire, con chiavi come nomi delle colonne e valori come dati
    :return: stringa "inserted" o "exists" a seconda dell'esito dell'operazione
    """
    conn = get_connection(db_name)
    c = conn.cursor()
    try:
        columns = ', '.join(data_dict.keys())
        placeholders = ', '.join(['?'] * len(data_dict))
        values = tuple(data_dict.values())
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        c.execute(query, values)
        conn.commit()
        return "inserted"
    except sqlite3.IntegrityError:
        return "exists"
    finally:
        conn.close()


def read_data(db_name, table_name):
    """
    Funzione per leggere dati all'interno del database
    - Connessione al database
    - Creazione di un cursore per eseguire le query
    - Query per leggere i dati dalla tabella
    - Recupera e memorizza in una variabile tutte le righe
    - Chiusura della connessione
    :param: db_name: nome del database
    :param: table_name: nome della tabella
    :return: righe della tabella
    """
    conn = get_connection(db_name)
    c = conn.cursor()
    query = f'SELECT * FROM {table_name}'
    c.execute(query)
    rows = c.fetchall()
    conn.close()
    return rows


def delete_data(db_name, table_name, conditions):
    """
    Funzione per eliminare dati all'interno del database
    - Connessione al database
    - Creazione di un cursore per eseguire le query
    - Costruzione dinamica della clausola WHERE in base alle condizioni fornite
    - Query per eliminare i dati nella tabella (usando segnaposto ? per evitare SQL injection)
    - Salvataggio dei cambiamenti e chiusura della connessione
    :param db_name: nome del database
    :param table_name: nome della tabella
    :param conditions: dizionario con le condizioni di eliminazione
    """
    conn = get_connection(db_name)
    c = conn.cursor()
    where_clause = ' AND '.join([f"{col} = ?" for col in conditions.keys()])
    values = tuple(conditions.values())
    query = f"DELETE FROM {table_name} WHERE {where_clause}"
    c.execute(query, values)
    conn.commit()
    conn.close()


def get_data(db_name, table_name, columns, conditions=None):
    """
    Funzione per leggere dati specifici da una tabella con condizioni facoltative
    - Connessione al database
    - Creazione di un cursore per eseguire le query
    - Costruzione dinamica della query con colonne selezionate e condizioni WHERE opzionali
    - Esecuzione sicura della query con parametri ? per prevenire SQL injection
    - Recupero delle righe
    - Chiusura della connessione
    :param db_name: nome del database
    :param table_name: nome della tabella
    :param columns: lista o stringa di colonne da leggere (es: ["id", "name"] o "id")
    :param conditions: dizionario di condizioni per la clausola WHERE. Se None, non viene applicato alcun filtro
    :return: righe selezionate della tabella (lista di tuple)
    """
    if isinstance(columns, str):
        columns = [columns]
    columns_str = ', '.join(columns)

    conn = get_connection(db_name)
    c = conn.cursor()

    if conditions:
        where_clause = ' AND '.join([f"{col} = ?" for col in conditions.keys()])
        values = tuple(conditions.values())
        query = f"SELECT {columns_str} FROM {table_name} WHERE {where_clause}"
        c.execute(query, values)
    else:
        query = f"SELECT {columns_str} FROM {table_name}"
        c.execute(query)

    rows = c.fetchall()
    conn.close()
    return rows


def close_connection(conn):
    """
    Funzione per chiudere la connessione al database (se usata)
    :param: conn: connessione da chiudere
    """
    conn.close()


def init_database():
    """
     Funzione per inizializzare il database
     - Crea le tabelle specificate se non esistono già
    """
    create_table("documents.db", '''
        CREATE TABLE IF NOT EXISTS receipts (
            Id INTEGER PRIMARY KEY AUTOINCREMENT,
            File_path TEXT UNIQUE,
            Upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    create_table("documents.db", '''
        CREATE TABLE IF NOT EXISTS extracted_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            receipt_id INTEGER NOT NULL,
            purchase_date DATE,
            purchase_time TIME,
            store_name TEXT,
            address TEXT,
            city TEXT,
            country TEXT,
            total_price REAL CHECK (total_price >= 0),
            total_currency TEXT CHECK (LENGTH(total_currency) = 3),
            payment_method TEXT,
            FOREIGN KEY(receipt_id) REFERENCES receipts(Id) ON DELETE CASCADE
        )
    ''')

    create_table("documents.db", '''
        CREATE TABLE IF NOT EXISTS receipt_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            extracted_data_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            price REAL CHECK (price >= 0),
            currency TEXT CHECK (LENGTH(currency) = 3),
            discount_percent REAL DEFAULT NULL CHECK (discount_percent >= 0 AND discount_percent <= 100),
            discount_value REAL DEFAULT NULL CHECK (discount_value >= 0),
            FOREIGN KEY(extracted_data_id) REFERENCES extracted_data(id) ON DELETE CASCADE
        )
    ''')
