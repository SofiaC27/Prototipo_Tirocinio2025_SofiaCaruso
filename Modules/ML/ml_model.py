import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew


from Modules.ML.ml_dataset import generate_dataset


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
    print("Prime righe del dataset:")
    print(df.head(), "\n")

    print("Dimensioni (righe, colonne):")
    print(df.shape, "\n")

    print("Nomi delle colonne:")
    print(df.columns.tolist(), "\n")

    print("Tipi di dato:")
    print(df.dtypes, "\n")

    print("Statistiche descrittive:")
    print(df.describe(include='all'), "\n")

    print("Info generale e valori non nulli:")
    print(df.info())


def handle_missing_values(df, target_col="next_week_spending"):
    """
    Funzione che gestisce i valori mancanti nel dataset:
    - Stampa la presenza di valori NaN per ogni colonna
    - Rimuove le righe dove la variabile target è NaN
    - Riempie con la media i NaN presenti in colonne numeriche diverse dalla variabile target
    :param df: dataset originale
    :param target_col: nome della colonna target
    :return: dataset pulito
    """
    # Visualizza la presenza di valori mancanti
    print("Ci sono valori mancanti?\n")
    print(df.isnull().any())
    print()

    # Seleziona colonne che contengono NaN
    missing_cols = df.columns[df.isnull().any()]

    # Gestisce i NaN solo se presenti
    for col in missing_cols:
        if col == target_col:
            # Rimuove righe con target mancante
            df = df[df[target_col].notna()].copy()
            continue

        # Riempie solo le colonne numeriche
        if pd.api.types.is_numeric_dtype(df[col]):
            mean_val = df[col].mean(skipna=True)
            df[col].fillna(mean_val, inplace=True)

    return df


def analyze_distributions(df, columns):
    """
    Funzione che analizza e visualizza la distribuzione delle variabili numeriche specificate:
    - Calcola l'asimmetria (skewness) per ogni colonna specificata
    - Visualizza l'istogramma e la stima della densità (KDE) per ogni colonna
    - Mostra i grafici uno per uno per facilitare l'interpretazione visiva
    :param df: dataset da analizzare
    :param columns: lista dei nomi di colonne numeriche da analizzare
    """
    print("\nAnalisi delle distribuzioni:")
    for col in columns:
        print(f"\nColonna: {col}")
        skewness = skew(df[col].dropna())
        print(f"Asimmetria (skewness): {skewness:.2f}")

        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=15, color="blue")
        plt.title(f"Distribuzione - {col}")
        plt.xlabel(col)
        plt.ylabel("Frequenza")
        plt.tight_layout()
        plt.show()


def detect_outliers_iqr(df, columns, show_plots=True):
    """
    Funzione che identifica outliers nei dati numerici tramite l'Interquartile Range (IQR):
    - Calcola il primo e il terzo quartile per ciascuna colonna numerica indicata
    - Usa il range interquartile per definire soglie oltre le quali i valori vengono considerati outliers
    - Stampa il numero di outliers trovati per colonna e, se richiesto, mostra i boxplot per visualizzarli
    :param df: dataset originale
    :param columns: lista delle colonne da analizzare
    :param show_plots: flag booleano per mostrare i boxplot (default=True)
    :return: dizionario riepilogativo con il conteggio degli outliers per ciascuna variabile
    """
    print("\nAnalisi outliers (IQR):")
    outlier_summary = {}

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]
        n_outliers = outliers.shape[0]
        outlier_summary[col] = n_outliers

        print(f"{col}: {n_outliers} outliers trovati (IQR = {IQR:.2f})")

        if show_plots:
            plt.figure(figsize=(5, 3))
            sns.boxplot(x=df[col], color="red")
            plt.title(f"Boxplot - {col}")
            plt.tight_layout()
            plt.show()

    return outlier_summary


warnings.filterwarnings("ignore")
df = generate_dataset()

# EDA #
print('EDA:\n')

# Ispezione iniziale del dataset
inspect_dataset(df)
print('\n')
# Controlla se ci sono valori mancanti
df = handle_missing_values(df)
print('\n')

# Analisi delle distribuzioni e dei valori anomali (outliers)
columns = [
    'total_spending',
    'n_receipts',
    'avg_receipt_spending',
    'delta_spending',
    'three_week_trend',
    'next_week_spending'
]

analyze_distributions(df, columns)
print('\n')
outlier_summary = detect_outliers_iqr(df, columns)
print("Riepilogo outliers:", outlier_summary)
print('\n')
