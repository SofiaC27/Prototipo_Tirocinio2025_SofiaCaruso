import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew


sns.set(style="whitegrid")


def inspect_dataset(df):
    """
    Funzione che esegue un’ispezione iniziale sul dataset:
    - Visualizza le prime righe
    - Mostra forma, colonne, tipi di dato
    - Calcola statistiche descrittive
    - Restituisce informazioni generali
    :param df: dataset da analizzare
    """
    print("-Prime righe del dataset:")
    print(df.head(), "\n")

    print("-Dimensioni (righe, colonne):")
    print(df.shape, "\n")

    print("-Nomi delle colonne:")
    print(df.columns.tolist(), "\n")

    print("-Tipi di dato:")
    print(df.dtypes, "\n")

    print("-Statistiche descrittive:")
    print(df.describe(include='all'), "\n")

    print("-Info generale e valori non nulli:")
    print(df.info())


def handle_missing_values(df, target_col="is_outlier"):
    """
    Funzione che gestisce i valori mancanti nel dataset:
    - Stampa la presenza di valori NaN per ogni colonna
    - Rimuove le righe dove la variabile target è NaN
    - Riempie con la media i NaN presenti in colonne numeriche diverse dalla variabile target
    - Stampa un messaggio per segnalare la presenza di eventuali valori mancanti e in caso spiega
      come li ha gestiti
    :param df: dataset originale
    :param target_col: nome della colonna target
    :return: dataset pulito
    """
    # Visualizza la presenza di valori mancanti
    print("-Ci sono valori mancanti?\n")
    print(df.isnull().any())
    print()

    # Seleziona colonne che contengono NaN
    missing_cols = df.columns[df.isnull().any()]
    managed_missing = False  # flag per tracciare se sono stati gestiti NaN

    # Gestisce i NaN solo se presenti
    for col in missing_cols:
        if col == target_col:
            # Rimuove righe con target mancante
            df = df[df[target_col].notna()].copy()
            managed_missing = True
            continue

        # Riempie solo le colonne numeriche
        if pd.api.types.is_numeric_dtype(df[col]):
            mean_val = df[col].mean(skipna=True)
            df[col].fillna(mean_val, inplace=True)
            managed_missing = True

    if managed_missing:
        print("Ci sono dei valori mancanti: sono state rimosse le righe con valori mancanti nella variabile "
              "target e riempiti con la media i valori mancanti nelle colonne numeriche")
    else:
        print("Non ci sono valori mancanti")

    return df


def plot_correlation_matrix(df, target_col="is_outlier"):
    """
    Funzione che calcola e visualizza la matrice di correlazione tra le variabili numeriche del dataset:
    - Calcola la correlazione tra coppie di variabili numeriche con il metodo di Pearson
    - Mostra una heatmap con colorazione basata sul valore della correlazione
    - Annota ogni cella della heatmap con il valore numerico della correlazione
    - Ottimizza il layout del grafico per migliorare la leggibilità
    :param df: dataset da analizzare
    :param target_col: nome della colonna target
    :return: la matrice di correlazione
    """
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    correlation_matrix = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matrice di Correlazione")
    plt.tight_layout()
    plt.show()

    return correlation_matrix


def plot_seasonal_outlier_rate(df):
    """
    Funzione che visualizza la frequenza di outlier per ciascuna stagione:
    - Raggruppa per stagione
    - Calcola la media della colonna target 'is_outlier'
    - Mostra i risultati con un grafico a barre
    :param df: dataset da analizzare
    """
    outlier_by_season = df.groupby("season")["is_outlier"].mean().sort_values()

    outlier_by_season.plot(kind="bar", color="green", figsize=(10, 8))
    plt.title("Frequenza di Outlier per Stagione")
    plt.xlabel("Stagione")
    plt.ylabel("Tasso di Outlier")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def plot_outlier_by_holiday(df):
    """
    Funzione che mostra la frequenza di outlier in date festive e non festive:
    - Confronta la colonna binaria 'is_holiday_week' rispetto a 'is_outlier'
    - Mostra la percentuale di outlier con un grafico a barre
    :param df: dataset da analizzare
    """
    outlier_by_holiday = df.groupby("is_holiday")["is_outlier"].mean()

    outlier_by_holiday.index = ["Data Non Festiva", "Data Festiva"]
    outlier_by_holiday.plot(kind="bar", color="blue", figsize=(10, 8))
    plt.title("Frequenza di Outlier: Festiva vs Non Festiva")
    plt.ylabel("Tasso di Outlier")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
