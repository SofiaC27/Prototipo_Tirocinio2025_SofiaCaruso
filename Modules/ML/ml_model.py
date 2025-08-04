import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


from Modules.ML.ml_dataset import generate_dataset
from Modules.ML.ml_eda import (inspect_dataset, handle_missing_values,plot_correlation_matrix,
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


'''
# Codifica le feature categoriche in numeriche
df = pd.get_dummies(df, columns=["season"], drop_first=True)


# Matrice di correlazione
corr_matrix = plot_correlation_matrix(df)
# Ordina le feature per correlazione col target
target_correlations = corr_matrix["next_week_spending"].drop("next_week_spending")
target_correlations_sorted = target_correlations.abs().sort_values(ascending=False)
print("Correlazione con il target:")
print(target_correlations_sorted)

# Feature Selection
selected_features = target_correlations[target_correlations.abs() > 0.3].index.tolist()
print("\nFeature selezionate (correlazione > 0.3):")
print(selected_features)
print('\n')

# Selezione delle feature (X) e assegnazione del target (y)
y = df["next_week_spending"]  # target

df_filtered = df[selected_features]
X = df_filtered.values  # features


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
        grid = GridSearchCV(estimator=model, param_grid=hparameters, scoring=scoring, refit='MSE', cv=cv)
        grid.fit(X_tr_transf, y_tr)
        trained_models.append((model_name, grid.best_estimator_))
        validation_performance.append(grid.best_score_)
        print('I valori migliori degli iperparametri sono: ', grid.best_params_)
        print('Metriche di validazione:')
        print('MSE:', grid.cv_results_['mean_test_MSE'][grid.best_index_])
        print('MAE:', grid.cv_results_['mean_test_MAE'][grid.best_index_])
        print('R2 :', grid.cv_results_['mean_test_R2'][grid.best_index_])


# Ensemble
print('\n  Random Forest Regressor')

rf_model = RandomForestRegressor(random_state=0)
rf_param_grid = {
    'n_estimators': [30, 50, 80],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 4, 6]
}

rf_grid = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, scoring=scoring, refit='MSE', cv=cv)
rf_grid.fit(X_tr_transf, y_tr)
trained_models.append(('RandomForest', rf_grid.best_estimator_))
validation_performance.append(rf_grid.best_score_)

print('\n I valori migliori degli iperparametri sono:', rf_grid.best_params_)
print('Metriche di validazione:')
print('MSE:', rf_grid.cv_results_['mean_test_MSE'][rf_grid.best_index_])
print('MAE:', rf_grid.cv_results_['mean_test_MAE'][rf_grid.best_index_])
print('R2 :', rf_grid.cv_results_['mean_test_R2'][rf_grid.best_index_])
'''
