# 1. IMPORT REQUIRED LIBRARIES
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

# 2. LOAD AND SPLIT DATA
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# 3. DEFINE HYPERPARAMETER GRID
parameters = {
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 200],
    "max_depth": [3, 5, -1]  # -1 MEANS INFINITE DEPTH IN LIGHTGBM
}

# 4. INITIALIZE MODEL (SILENT MODE) AND GRID SEARCH
model_lgbm = LGBMClassifier(random_state=42, verbose=-1)
# cv=5 MEANS 5-FOLD CROSS VALIDATION, n_jobs=-1 USES ALL CPU CORES
grid_search = GridSearchCV(estimator=model_lgbm, param_grid=parameters, cv=5, n_jobs=-1)

# 5. FIT GRID SEARCH (TRAINS 18 COMBINATIONS x 5 FOLDS = 90 MODELS)
grid_search.fit(X_train, y_train)

# 6. EXTRACT AND PRINT BEST PARAMETERS
print(f"Best Hyperparameters: {grid_search.best_params_}")

# 7. MAKE PREDICTIONS WITH THE BEST MODEL AND EVALUATE
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

print(f"Success of Optimized LGBM: {accuracy_score(y_test, predictions):.4f}")