import os
import json
import pandas as pd
from datetime import datetime
import holidays
from sklearn.ensemble import IsolationForest

# Calcola il percorso assoluto della root del progetto, risalendo di due livelli dalla cartella dello script corrente
# Costruisce il percorso completo della cartella 'Extracted_JSON' all’interno della root del progetto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXTRACTED_JSON_DIR = os.path.join(PROJECT_ROOT, "Extracted_JSON")


def load_receipts_json(json_dir=EXTRACTED_JSON_DIR):
    """
    Funzione per caricare tutti i file JSON dalla cartella in cui sono salvati:
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


def assign_season(date):
    """
    Funzione che assegna una stagione approssimativa in base alla settimana ISO dell'anno:
    corrispondente alla data fornita:
    - Primavera: settimane da 10 a 22
    - Estate: settimane da 23 a 35
    - Autunno: settimane da 36 a 48
    - Inverno: tutte le altre settimane
    :param date: data da cui estraare la settimana per assegnare la stagione
    :return: stringa che rappresenta la stagione ("primavera", "estate", "autunno", "inverno")
    """
    week = date.isocalendar().week
    if 10 <= week <= 22:
        return "primavera"
    elif 23 <= week <= 35:
        return "estate"
    elif 36 <= week <= 48:
        return "autunno"
    else:
        return "inverno"


def is_holiday(date, country_code="IT"):
    """
    Funzione che verifica se una data specifica corrisponde a una festività nel paese indicato:
    - Utilizza la libreria 'holidays' per identificare le date festive
    :param date: data da valutare come festiva o no
    :param country_code: codice paese (default "IT" per Italia)
    :return: 1 se la data è festiva, 0 altrimenti
    """
    holiday_calendar = holidays.country_holidays(country_code)
    if date in holiday_calendar:
        return 1  # La data è una festività
    else:
        return 0  # La data NON è una festività


def create_dataset_from_receipts():
    """
    Funzione per creare un dataset a partire dai file JSON degli scontrini:
    - Carica i dati grezzi e verifica che ogni ricevuta contenga una data valida
    - Per ciascuno scontrino: calcola giorno della settimana, mese, stagione e festività; estrae la
      spesa totale e il numero di articoli; calcola la spesa media per articolo
    - Arrotonda i valori numerici a due cifre decimali per maggiore coerenza e leggibilità
    - Salva le informazioni in una lista di record e converte la lista in un DataFrame pandas
    - Verifica se il DataFrame è vuoto e in caso restituisce un messaggio
    :return: DataFrame con le informazioni di ciascuno scontrino
    """
    raw_data = load_receipts_json()
    records = []

    for receipt in raw_data:
        try:
            date_str = receipt.get("data")
            if not date_str:
                continue

            date = datetime.strptime(date_str, "%Y-%m-%d").date()

            val = receipt.get("prezzo_totale", {}).get("valore")
            total_price = float(val) if val is not None else 0.0

            items = receipt.get("lista_articoli", [])
            n_items = sum([
                int(q) if q is not None else 0
                for q in [item.get("quantita") for item in items]
            ])

            spending_per_item = total_price / n_items if n_items else 0.0

            records.append({
                "date": date,
                "day_of_week": date.weekday(),
                "month": date.month,
                "season": assign_season(date),
                "is_holiday": is_holiday(date),
                "total_price": round(total_price, 2),
                "n_items": n_items,
                "spending_per_item": round(spending_per_item, 2)
            })

        except Exception as e:
            print("Errore in uno scontrino:", e)

    df = pd.DataFrame(records)

    if df.empty:
        print("Nessun dato valido trovato")
        return pd.DataFrame()

    return df


def label_outliers(df):
    """
    Funzione per etichettare gli scontrini anomali (outlier) all'interno di un dataset:
    - Utilizza l'algoritmo Isolation Forest per rilevare anomalie nei dati
    - Considera tre variabili per l'identificazione: total_price, n_items, spending_per_item
    - Assegna 1 agli scontrini anomali e 0 a quelli regolari
    - Aggiunge la colonna 'is_outlier' al DataFrame
    :param df: DataFrame contenente i dati da etichettare
    :return: DataFrame aggiornato con la colonna target 'is_outlier'
    """
    model = IsolationForest(contamination='auto', random_state=0)
    df["is_outlier"] = model.fit_predict(df[["total_price", "n_items", "spending_per_item"]])
    df["is_outlier"] = df["is_outlier"].apply(lambda x: 1 if x == -1 else 0)
    return df


def generate_dataset():
    """
    Funzione per generare un dataset completo con informazioni sugli scontrini e rilevamento outlier:
    - Crea un DataFrame iniziale e applica delle label per marcare gli scontrini anomali
    :return: DataFrame finale
    """
    df = create_dataset_from_receipts()
    df = label_outliers(df)
    return df
