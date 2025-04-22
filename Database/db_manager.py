import sqlite3


def create_table():
    """
    Funzione per creare la tabella 'scontrini'
    - Connessione al database 'documenti.db'
    - Creazione di un cursore per eseguire le query
    - Query per creare (solo una volta) la tabella se non esiste già
    - Salvataggio dei cambiamenti e chiusura della connessione
    """
    conn = sqlite3.connect("documenti.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS scontrini (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome_file TEXT,
            data_caricamento TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def get_connection():
    """
    Funzione per connettersi al database
    :return: connessione al database 'documenti.db'
    """
    return sqlite3.connect("documenti.db")  # Return a connection to the database


def insert_data(file_name):
    """
    Funzione per inserire dati all'interno del database
    - Connessione al database 'documenti.db'
    - Creazione di un cursore per eseguire le query
    - Query per inserire i dati nella tabella 'scontrini'
    - Il ? è un segnaposto per valori da inserire in sicurezza tramite parametri, evitando SQL injection
    - La , alla fine indica che (file_name,) è una tupla con un singolo elemento
    - Salvataggio dei cambiamenti e chiusura della connessione
    :param file_name: file da inserire
    """
    conn = get_connection()
    c = conn.cursor()
    c.execute('INSERT INTO scontrini (nome_file) VALUES (?)', (file_name,))
    conn.commit()
    conn.close()


def read_data():
    """
    Funzione per leggere dati all'interno del database
    - Connessione al database 'documenti.db'
    - Creazione di un cursore per eseguire le query
    - Query per leggere i dati dalla tabella 'scontrini'
    - Recupera e memorizza in una variabile tutte le righe
    - Chiusura della connessione
    :return: righe della tabella
    """
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM scontrini')
    rows = c.fetchall()
    conn.close()
    return rows


def delete_data(file_name):
    """
    Funzione per eliminare dati all'interno del database
    - Connessione al database 'documenti.db'
    - Creazione di un cursore per eseguire le query
    - Query per eliminare i dati nella tabella 'scontrini'
    - Il ? è un segnaposto per valori da eliminare in sicurezza tramite parametri, evitando SQL injection
    - La , alla fine indica che (file_name,) è una tupla con un singolo elemento
    - Salvataggio dei cambiamenti e chiusura della connessione
    :param file_name: file da eliminare
    """
    conn = get_connection()
    c = conn.cursor()
    c.execute('DELETE FROM scontrini WHERE nome_file = ?', (file_name,))
    conn.commit()
    conn.close()


# Function to close the connection (if used)
def close_connection(conn):
    """
    Funzione per chiudere la connessione al database 'documenti.db'
    :param: conn: connessione da chiudere
    """
    conn.close()


create_table()  # Chiama la funzione per inizializzare la tabella quando lo script viene eseguito
