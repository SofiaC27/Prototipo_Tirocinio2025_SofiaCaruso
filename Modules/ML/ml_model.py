import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


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

# Training #
print('Training:\n')

# Divide il dataset: 80% per il training e 20% per il testing
test_split_ratio = 0.2
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=test_split_ratio, random_state=0)

# Scalamento dei dati
scaler = StandardScaler()
scaler.fit(X_tr)
X_tr_transf = scaler.transform(X_tr)
n_features = X_tr_transf.shape[1]

# Definizione dei modelli
models = [
    LinearRegression(),  # Linear Regression
    SGDRegressor(random_state=0),  # Stochastic Gradient Descent Regressor
    KNeighborsRegressor(),  # K-NN Regressor
    DecisionTreeRegressor(random_state=0),  # Decision Tree Regressor
    SVR()  # SVR
]

models_names = [
    'Linear Regression',
    'SGD Regressor',
    'K-NN Regressor',
    'DT Regressor',
    'SVR'
]

models_hyperparameters = [
    {},  # LinearRegression non ha iperparametri rilevanti da ottimizzare
    {  # SGDRegressor
        'penalty': ['l2', 'l1'],
        'alpha': [1e-5, 1e-4, 1e-3, 1e-2],
        'learning_rate': ['constant', 'optimal', 'invscaling'],
        'eta0': [0.001, 0.01, 0.1, 1.0],
    },
    {  # K-NN Regressor
        'n_neighbors': list(range(1, 10, 2)),
        'weights': ['uniform', 'distance']
    },
    {  # Decision Tree Regressor
        'max_depth': [None, 3, 5, 10, 15],
        'min_samples_split': [2, 5, 10, 20],
    },
    {  # SVR
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf']
    }
]

cv = KFold(n_splits=5, shuffle=True, random_state=0)
scoring = {
    'MSE': 'neg_mean_squared_error',
    'MAE': 'neg_mean_absolute_error',
    'R2': 'r2'
}
trained_models = []
validation_performance = []

for model, model_name, hparameters in zip(models, models_names, models_hyperparameters):
        print('\n ', model_name)
        clf = GridSearchCV(estimator=model, param_grid=hparameters, scoring=scoring, refit='MSE', cv=cv)
        clf.fit(X_tr_transf, y_tr)
        trained_models.append((model_name, clf.best_estimator_))
        validation_performance.append(clf.best_score_)
        print('I valori migliori degli iperparametri sono: ', clf.best_params_)
        print('Metriche di validazione:')
        print('MSE:', clf.cv_results_['mean_test_MSE'][clf.best_index_])
        print('MAE:', clf.cv_results_['mean_test_MAE'][clf.best_index_])
        print('R2 :', clf.cv_results_['mean_test_R2'][clf.best_index_])
