import warnings
import pandas as pd
import numpy as np


from Modules.ML.ml_dataset import generate_dataset
from Modules.ML.ml_eda import inspect_dataset, handle_missing_values, plot_correlation_matrix

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
# Matrice di correlazione
corr_matrix = plot_correlation_matrix(df)
print('\n')

# Selezione delle feature (X) e assegnazione del target (y)
y = df["next_week_spending"]  # target

# Rimuove le colonne con feature non necessarie
df = df.drop(columns=["next_week_spending", "year", "week"])
# Codifica le feature categoriche in numeriche
df = pd.get_dummies(df, columns=["season"], drop_first=True)

X = df.values  # features
