import warnings
import pandas as pd
import numpy as np

from Modules.ML.ml_dataset import generate_dataset


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
