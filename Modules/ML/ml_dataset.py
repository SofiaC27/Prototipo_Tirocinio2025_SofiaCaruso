import os
import json
import pandas as pd
from datetime import datetime

# Calcola il percorso assoluto della root del progetto, risalendo di due livelli dalla cartella dello script corrente
# Costruisce il percorso completo della cartella 'Extracted_JSON' all’interno della root del progetto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXTRACTED_JSON_DIR = os.path.join(PROJECT_ROOT, "Extracted_JSON")


def load_receipts_json(json_dir=EXTRACTED_JSON_DIR):
    """
    Funzione per caricare tutti i file JSON dalla cartella in cui sono salvati
    - Legge ogni file con estensione .json presente nella cartella specificata
    - Converte il contenuto di ciascun file in un dizionario Python
    - Ignora file non validi o con errori di decodifica JSON
    - Restituisce una lista contenente tutti i dizionari letti
    :param json_dir: percorso della cartella contenente i file JSON
    :return: lista di dizionari ottenuti dai file JSON validi
    """
    data = []
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(json_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    receipt = json.load(f)
                    data.append(receipt)
                except json.JSONDecodeError:
                    print(f"Errore nel file: {filename}")
    return data


def create_weekly_dataset():
    """
    Funzione per creare un dataset settimanale a partire dai file JSON degli scontrini
    - Carica i dati grezzi e verifica che ogni ricevuta contenga una data valida
    - Estrae anno e numero di settimana da ciascuna data di acquisto
    - Costruisce un elenco di record con data e importo speso per ciascuna settimana
    - Aggrega i dati per settimana calcolando spesa totale, numero di scontrini e spesa media per scontrino
    - Crea una colonna target che rappresenta la spesa totale prevista per la settimana successiva
    - Arrotonda i valori numerici a due cifre decimali per maggiore coerenza e leggibilità
    :return: DataFrame con aggregati settimanali e colonna target per regressione
    """
    raw_data = load_receipts_json()
    records = []

    for receipt in raw_data:
        try:
            date_str = receipt.get("data")
            total_price = float(receipt.get("prezzo_totale", {}).get("valore", 0))

            if not date_str:
                continue

            date = datetime.strptime(date_str, "%Y-%m-%d")
            year, week_num, _ = date.isocalendar()

            records.append({
                "year": year,
                "week": week_num,
                "total_price": total_price
            })
        except Exception as e:
            print("Errore in uno scontrino:", e)

    df = pd.DataFrame(records)

    if df.empty:
        print("Nessun dato valido trovato")
        return pd.DataFrame()

    # Aggrega per settimana
    weekly_df = df.groupby(["year", "week"]).agg(
        total_spending=("total_price", "sum"),
        n_receipts=("total_price", "count"),
        avg_receipt_spending=("total_price", "mean")
    ).reset_index()

    # Crea colonna target: spesa della settimana successiva
    weekly_df["next_week_spending"] = weekly_df["total_spending"].shift(-1)

    # Arrotonda tutte le colonne numeriche a due decimali per uniformità e leggibilità
    weekly_df = weekly_df.round(2)

    return weekly_df
