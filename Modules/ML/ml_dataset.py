import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import holidays

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
    - Calcola la spesa totale e il numero di articoli per scontrino
    - Costruisce un elenco di record con data, importo speso e numero di articoli per ciascuna settimana
    - Aggrega i dati per settimana sommando spesa e articoli, e contando il numero di scontrini
    - Costruisce un DataFrame con statistiche settimanali
    - Aggiunge una colonna target che rappresenta la spesa totale della settimana successiva
    - Arrotonda i valori numerici a due cifre decimali per maggiore coerenza e leggibilità
    :return: DataFrame con aggregati settimanali e colonna target per regressione
    """
    raw_data = load_receipts_json()
    records = []

    for receipt in raw_data:
        try:
            date_str = receipt.get("data")

            val = receipt.get("prezzo_totale", {}).get("valore")
            total_price = float(val) if val is not None else 0.0

            items = receipt.get("lista_articoli", [])
            n_items = sum([
                int(q) if q is not None else 0
                for q in [item.get("quantita") for item in items]
            ])

            if not date_str:
                continue

            date = datetime.strptime(date_str, "%Y-%m-%d")
            year, week_num, _ = date.isocalendar()

            records.append({
                "year": year,
                "week": week_num,
                "total_price": total_price,
                "total_items": n_items
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
        total_items=("total_items", "sum")
    ).reset_index()

    # Crea colonna target: spesa della settimana successiva
    weekly_df["next_week_spending"] = weekly_df["total_spending"].shift(-1)
    weekly_df = weekly_df.round(2)

    return weekly_df


def assign_season(week):
    """
    Funzione che assegna una stagione approssimativa all'intervallo settimanale fornito,
    basandosi sul numero ISO della settimana dell'anno:
    - Primavera: settimane da 10 a 22
    - Estate: settimane da 23 a 35
    - Autunno: settimane da 36 a 48
    - Inverno: tutte le altre settimane
    :param week: numero della settimana secondo lo standard ISO
    :return: stringa che rappresenta la stagione ("primavera", "estate", "autunno", "inverno")
    """
    if 10 <= week <= 22:
        return "primavera"
    elif 23 <= week <= 35:
        return "estate"
    elif 36 <= week <= 48:
        return "autunno"
    else:
        return "inverno"


def is_week_holiday(year, week, country_code="IT"):
    """
    Funzione che verifica se una settimana dell’anno contiene almeno una festività nel paese indicato
    - Controlla tutti e 7 i giorni della settimana ISO (da lunedì a domenica)
    - Utilizza la libreria 'holidays' per identificare le date festive
    :param year: anno di riferimento
    :param week: numero della settimana ISO
    :param country_code: codice paese (default "IT" per Italia)
    :return: 1 se almeno un giorno è festivo, 0 altrimenti
    """
    try:
        holiday_calendar = holidays.country_holidays(country_code)
        # Controlla tutti i giorni della settimana ISO (1 = lunedì, 7 = domenica)
        for d in range(1, 8):
            date = datetime.strptime(f"{year}-W{int(week):02d}-{d}", "%Y-W%W-%w")
            if date in holiday_calendar:
                return 1
    except ValueError:
        return 0
    return 0


def add_engineered_features(df):
    """
    Funzione per arricchire il dataset settimanale con feature ingegnerizzate
    - Controlla che il dataset non sia vuoto prima di elaborarlo
    - Aggiunge una codifica ciclica della settimana per rappresentarne la periodicità
    - Calcola la variazione percentuale della spesa rispetto alla settimana precedente
    - Inserisce una media mobile della spesa sulle ultime tre settimane come indicatore di trend
    - Deriva metriche settimanali di comportamento come spesa media per articolo, densità di scontrini e
      articoli medi per ricevuta
    - Assegna la stagione a ciascuna settimana e identifica quelle festive
    - Calcola la media di spesa stagionale e il rapporto rispetto alla spesa della settimana corrente
    - Calcola la media di spesa durante le settimane festive e il rapporto rispetto alla spesa corrente
    - Arrotonda tutti i valori numerici a due cifre decimali per maggiore coerenza e leggibilità
    :return: DataFrame arricchito con feature ingegnerizzate
    """
    if df.empty:
        return df

    # Codifica ciclica settimana
    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)

    # Delta % di spesa
    df["delta_spending_pct"] = df["total_spending"].pct_change().fillna(0)

    # Media mobile sulle ultime 3 settimane
    df["three_week_trend"] = df["total_spending"].rolling(window=3).mean()

    # Metriche settimanali
    df["avg_items_per_receipt"] = df["total_items"] / df["n_receipts"]
    df["receipt_density"] = df["n_receipts"] / 7
    df["spending_per_item"] = df["total_spending"] / df["total_items"]

    # Stagione e festività
    df["season"] = df["week"].apply(assign_season)
    df["is_holiday_week"] = df.apply(lambda row: is_week_holiday(row["year"], row["week"]), axis=1)

    # Media stagionale e confronto
    df["season_avg_spending"] = df.groupby("season")["total_spending"].transform("mean")
    df["spending_vs_season_avg"] = df["total_spending"] / df["season_avg_spending"]

    # Media festività e confronto
    holiday_avg = df[df["is_holiday_week"] == 1]["total_spending"].mean()
    df["spending_vs_holiday_avg"] = df["total_spending"] / holiday_avg

    # Rimuove la feature 'week' non più necessaria
    df.drop(columns=["week"], inplace=True)

    df = df.round(2)

    return df


def reorder_columns(df, target="next_week_spending"):
    """
    Funzione che riordina le colonne del dataset spostando la variabile target alla fine
    - Crea una lista che contiene tutte le colonne del DataFrame tranne quella indicata come target
    - Poi aggiunge il target alla fine della lista
    - Infine, riordina il DataFrame mettendo la colonna target come ultima
    - Utile per preparare il dataset per modelli ML o salvataggio pulito
    :param df: DataFrame da riordinare
    :param target: nome della colonna target da spostare in fondo (default "next_week_spending")
    :return: DataFrame con colonne riordinate
    """
    other_columns = []
    for col in df.columns:
        if col != target:
            other_columns.append(col)

    ordered_columns = other_columns + [target]
    df = df[ordered_columns]

    return df


def generate_dataset():
    """
    Funzione che genera il dataset finale pronto per l'utilizzo
    - Crea il dataset settimanale a partire dai file JSON
    - Aggiunge feature ingegnerizzate
    - Riordina le colonne portando la variabile target alla fine
    :return: DataFrame finale pronto per essere visualizzato o passato a modelli ML
    """
    df = create_weekly_dataset()

    if df.empty:
        print("Dataset vuoto, nessun dato da visualizzare.")
        return df

    df = add_engineered_features(df)
    df = reorder_columns(df)

    return df
