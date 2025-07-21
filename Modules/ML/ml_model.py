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
    print("-Ci sono valori mancanti?\n")
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
    print("\n-Analisi delle distribuzioni:")
    for col in columns:
        print(f"\nColonna: {col}")
        skewness = skew(df[col].dropna())
        print(f"Asimmetria (skewness): {skewness:.2f}")

        # plt.figure(figsize=(6, 4))
        # sns.histplot(df[col], kde=True, bins=15, color="blue")
        # plt.title(f"Distribuzione - {col}")
        # plt.xlabel(col)
        # plt.ylabel("Frequenza")
        # plt.tight_layout()
        # plt.show()


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
    print("\n-Analisi outliers (IQR):")
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

        # if show_plots:
            # plt.figure(figsize=(5, 3))
            # sns.boxplot(x=df[col], color="red")
            # plt.title(f"Boxplot - {col}")
            # plt.tight_layout()
            # plt.show()

    return outlier_summary


def plot_correlation_matrix(df):
    """
    Funzione che calcola e visualizza la matrice di correlazione tra le variabili numeriche del dataset:
    - Calcola la correlazione tra coppie di variabili numeriche con il metodo di Pearson
    - Mostra una heatmap con colorazione basata sul valore della correlazione
    - Annota ogni cella della heatmap con il valore numerico della correlazione
    - Ottimizza il layout del grafico per migliorare la leggibilità
    :param df: dataset da analizzare
    """
    correlation_matrix = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matrice di Correlazione")
    plt.tight_layout()
    plt.show()


def plot_spending_over_time(df):
    """
    Funzione che visualizza l’andamento della spesa totale nel tempo:
    - Utilizza la colonna "week" come variabile temporale sull’asse x
    - Visualizza la spesa totale ("total_spending") sull'asse y, usando un grafico con punti marcati
    - Aggiunge titolo, etichette e ottimizza la disposizione degli elementi grafici
    :param df: dataset contenente le colonne "week" e "total_spending"
    """
    df.plot(x="week", y="total_spending", marker='o', figsize=(10, 5), title="Spesa Totale nel Tempo")
    plt.ylabel("Totale Spesa")
    plt.xlabel("Settimana")
    plt.tight_layout()
    plt.show()


def plot_seasonal_spending(df):
    """
    Funzione che visualizza la spesa media aggregata per ciascuna stagione:
    - Raggruppa i dati in base alla stagione indicata nella colonna "season"
    - Calcola la media della spesa totale ("total_spending") per ogni stagione
    - Visualizza i valori medi con un grafico a barre e aggiunge etichette e titolo
    - Ottimizza la disposizione degli elementi grafici
    :param df: dataset contenente le colonne "season" e "total_spending"
    """
    df.groupby("season")["total_spending"].mean().plot(
        kind="bar", title="Spesa Media per Stagione"
    )
    plt.ylabel("Totale Spesa")
    plt.xlabel("Stagione")
    plt.tight_layout()
    plt.show()


def plot_holiday_impact(df):
    """
    Funzione che visualizza la distribuzione della spesa totale a confronto tra settimane festive e non festive:
    - Suddivide il dataset in due categorie basate sulla variabile binaria "is_holiday_week"
    - Per ciascun gruppo, disegna un boxplot della variabile "total_spending"
    - Evidenzia differenze di mediana, variabilità e outliers tra i due contesti
    - Ottimizza graficamente la disposizione degli elementi per facilitarne la lettura
    :param df: dataset contenente le colonne "is_holiday_week" e "total_spending"
    """
    sns.boxplot(x="is_holiday_week", y="total_spending", data=df)
    plt.title("Spesa in Settimana Festiva vs Non Festiva")
    plt.xlabel("Settimana Festiva")
    plt.ylabel("Totale Spesa")
    plt.tight_layout()
    plt.show()


def pairplot_features(df, columns):
    """
    Funzione che crea una griglia di scatterplot e istogrammi (pairplot) per visualizzare le relazioni tra
    più feature numeriche:
    - Disegna scatterplot per tutte le combinazioni possibili tra le colonne specificate
    - Mostra le distribuzioni univariate in diagonale mediante istogrammi
    - Aggiunge un titolo al grafico e ottimizza la disposizione degli elementi per migliorare la leggibilità
    :param df: dataset contenente le colonne numeriche da analizzare
    :param columns: lista di nomi delle variabili numeriche da includere nella griglia di pairplot
    """
    sns.pairplot(df[columns])
    plt.suptitle("Analisi Multivariata - Pairplot", y=1.02)
    plt.tight_layout()
    plt.show()


warnings.filterwarnings("ignore")
df = generate_dataset()

# EDA #
print('EDA:\n')

# Colonne numeriche da analizzare
columns = [
    'total_spending',
    'n_receipts',
    'avg_receipt_spending',
    'delta_spending',
    'three_week_trend',
    'next_week_spending'
]

# Ispezione iniziale del dataset
inspect_dataset(df)
print('\n')
# Controlla se ci sono valori mancanti
df = handle_missing_values(df)
print('\n')

# Analisi delle distribuzioni e dei valori anomali (outliers)
analyze_distributions(df, columns)
print('\n')
outlier_summary = detect_outliers_iqr(df, columns)
print("Riepilogo outliers:", outlier_summary)
print('\n')

print("-Interpretazione analisi distribuzioni e outlier:\n"
      "Le variabili analizzate mostrano una moderata asimmetria (skewness < 3) e un numero limitato di outlier.\n"
      "Non si osservano anomalie significative tali da richiedere trasformazioni dei dati o rimozione di righe.\n"
      "I valori sono considerati accettabili per i modelli ML previsti, quindi i dati vengono mantenuti senza"
      " modifiche.\n")

# Analisi multivariata, matrice di correlazione e visualizzazioni grafiche dei dati
plot_correlation_matrix(df)
plot_spending_over_time(df)
plot_seasonal_spending(df)
plot_holiday_impact(df)
pairplot_features(df, columns)
