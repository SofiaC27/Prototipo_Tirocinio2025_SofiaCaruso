import sqlite3

# Function to create the table (only once)
def create_table():
    conn = sqlite3.connect("documenti.db")  # Connect to the database
    c = conn.cursor()  # Create a cursor to execute queries
    c.execute('''
        CREATE TABLE IF NOT EXISTS scontrini (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome_file TEXT,
            data_caricamento TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')  # Create the table if it does not exist
    conn.commit()  # Commit the changes
    conn.close()  # Close the connection

# Function to obtain a connection to the database
def get_connection():
    return sqlite3.connect("documenti.db")  # Return a connection to the database

# Function to insert data into the database
def insert_data(file_name):
    conn = get_connection()  # Get a connection to the database
    c = conn.cursor()  # Create a cursor to execute queries
    c.execute('INSERT INTO scontrini (nome_file) VALUES (?)', (file_name,))  # Insert data into the table
    conn.commit()  # Commit the changes
    conn.close()  # Close the connection

# Function to read data from the database
def read_data():
    conn = get_connection()  # Get a connection to the database
    c = conn.cursor()  # Create a cursor to execute queries
    c.execute('SELECT * FROM scontrini')  # Retrieve all rows from the table
    rows = c.fetchall()  # Fetch all the rows
    conn.close()  # Close the connection
    return rows  # Return the rows


# Function to close the connection (if used)
def close_connection(conn):
    conn.close()  # Close the given connection

# Create the table at startup
create_table()  # Initialize the database table when the script runs
