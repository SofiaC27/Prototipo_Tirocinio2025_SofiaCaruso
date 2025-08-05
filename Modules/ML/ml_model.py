import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


from Modules.ML.ml_dataset import generate_dataset
from Modules.ML.ml_eda import (inspect_dataset, handle_missing_values, plot_correlation_matrix,
                               plot_seasonal_outlier_rate, plot_outlier_by_holiday)

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
# Visualizza la matrice di correlazione tra le feature numeriche
plot_correlation_matrix(df)
# Visualizza dei grafici a barre che mostrano la frequenza degli outlier in base alla stagione o alle festività
plot_seasonal_outlier_rate(df)
plot_outlier_by_holiday(df)


# Codifica le feature categoriche in numeriche
df = pd.get_dummies(df, columns=["season"], drop_first=True)

# Selezione delle feature (X) e assegnazione del target (y)
y = df['is_outlier']  # target
X = df.drop(['is_outlier', 'date'], axis=1).values  # features


# Controlla lo sbilanciamento delle classi della variabile target
class_counts = y.value_counts().sort_index()

labels = ['Normale (0)', 'Outlier (1)']
colors = ['red', 'yellow']

plt.figure(figsize=(8, 5))
plt.bar(class_counts.index, class_counts.values, color=colors)
plt.xticks(class_counts.index, labels)
plt.xlabel('Classe is_outlier')
plt.ylabel('Numero di sample')
plt.title('Distribuzione delle classi (is_outlier)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

print("Distribuzione percentuale delle classi:\n")
print(y.value_counts(normalize=True).sort_index())
print('\n')


# Training #
print('Training:\n')

# Divide il dataset: 80% per il training e 20% per il testing
test_split_ratio = 0.2
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=test_split_ratio, random_state=0, stratify=y)

# Scalamento dei dati
scaler = StandardScaler()
scaler.fit(X_tr)
X_tr_transf = scaler.transform(X_tr)
print('Numero di training samples =', X_tr_transf.shape[0])


# Definizione dei modelli
models = [
    LogisticRegression(solver='liblinear', class_weight='balanced', random_state=0),  # Logistic Regression
    KNeighborsClassifier(),  # K-NN
    SVC(class_weight='balanced', random_state=0),  # SVM
    DecisionTreeClassifier(class_weight='balanced', random_state=0),  # Decision Tree
]

models_names = [
    'Logistic Regression',
    'K-NN',
    'SVM',
    'Decision Tree',
]

models_hyperparameters = [
    {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10]},  # Logistic Regression
    {'n_neighbors': list(range(1, 10, 2))},  # K-NN
    {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': [0.001, 0.0001]},  # SVM
    {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5]},  # Decision Tree
]


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
trained_models = []
validation_performance = []

for model, model_name, hparameters in zip(models, models_names, models_hyperparameters):
    print('\n ', model_name)
    clf = GridSearchCV(estimator=model, param_grid=hparameters, scoring='balanced_accuracy', cv=cv)
    clf.fit(X_tr_transf, y_tr)
    trained_models.append((model_name, clf.best_estimator_))
    print('I valori migliori degli iperparametri sono:  ', clf.best_params_)
    print('Balanced Accuracy:  ', clf.best_score_)
    validation_performance.append(clf.best_score_)


# Scelta finale del modello
best_model_index = np.argmax(validation_performance)
final_model = trained_models[best_model_index][1]
print('\nHo scelto come miglior modello : ', trained_models[best_model_index][0])

# Training finale con tutto il dataset di training
final_model.fit(X_tr_transf, y_tr)
print('\n')


# Testing #
print('Testing:\n')

# Scalamento dei dati
X_ts_transf = scaler.transform(X_ts)

# Risultati finali
y_pred = final_model.predict(X_ts_transf)
test_balanced_accuracy = balanced_accuracy_score(y_ts, y_pred)

print('Risultati finali del testing\n')
print('Numero di testing samples =', X_ts_transf.shape[0])
print('Balanced Accuracy: ', test_balanced_accuracy)
print('\n')


mod_names = [name for name, model in trained_models]

colors2 = ['red', 'blue', 'green', 'orange']
plt.figure(figsize=(8, 5))
plt.bar(mod_names, validation_performance, color=colors2)
plt.ylim([0, 1])
plt.ylabel('Balanced Accuracy')
plt.title('Balanced Accuracy dei modelli (Cross-Validation)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.text(0.5, -0.1, f'Performance finale sul testing set: {test_balanced_accuracy * 100:.1f}% Balanced Accuracy',
         fontsize=12, ha='center', transform=plt.gca().transAxes)
plt.show()
