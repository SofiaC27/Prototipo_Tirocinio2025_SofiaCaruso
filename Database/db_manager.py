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


def insert_data(db_name, table_name, file_name):
    """
    Funzione per inserire dati all'interno del database
    - Connessione al database
    - Creazione di un cursore per eseguire le query
    - Query per inserire i dati nella tabella
    - Il ? è un segnaposto per valori da inserire in sicurezza tramite parametri, evitando SQL injection
    - La , alla fine indica che (file_name,) è una tupla con un singolo elemento
    - Salvataggio dei cambiamenti e chiusura della connessione
    :param: db_name: nome del database
    :param: table_name: nome della tabella
    :param file_name: file da inserire
    """
    conn = get_connection(db_name)
    c = conn.cursor()
    query = f'INSERT INTO {table_name} (file_name) VALUES (?)'
    c.execute(query, (file_name,))
    conn.commit()
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


def delete_data(db_name, table_name, file_name):
    """
    Funzione per eliminare dati all'interno del database
    - Connessione al database
    - Creazione di un cursore per eseguire le query
    - Query per eliminare i dati nella tabella
    - Il ? è un segnaposto per valori da eliminare in sicurezza tramite parametri, evitando SQL injection
    - La , alla fine indica che (file_name,) è una tupla con un singolo elemento
    - Salvataggio dei cambiamenti e chiusura della connessione
    :param: db_name: nome del database
    :param: table_name: nome della tabella
    :param file_name: file da eliminare
    """
    conn = get_connection(db_name)
    c = conn.cursor()
    query = f'DELETE FROM {table_name} WHERE file_name = ?'
    c.execute(query, (file_name,))
    conn.commit()
    conn.close()


def close_connection(conn):
    """
    Funzione per chiudere la connessione al database (se usata)
    :param: conn: connessione da chiudere
    """
    conn.close()


create_table("documents.db", '''
        CREATE TABLE IF NOT EXISTS receipts (
            Id INTEGER PRIMARY KEY AUTOINCREMENT,
            File_path TEXT,
            Upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
