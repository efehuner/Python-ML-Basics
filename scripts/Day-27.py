# 1. IMPORT REQUIRED LIBRARIES
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 2. LOAD AND SPLIT DATA
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# 3. INITIALIZE AND TRAIN XGBOOST MODEL
model_xgb = XGBClassifier(eval_metric="logloss", random_state=42)
model_xgb.fit(X_train, y_train)

# 4. MAKE PREDICTIONS ON TEST DATA
prediction_xgb = model_xgb.predict(X_test)

# 5. EVALUATE SUCCESS AND PRINT FEATURE IMPORTANCES
print(f"Success of XGB: {accuracy_score(y_test, prediction_xgb):.4f}")

print(f"Importance of features: \n{model_xgb.feature_importances_}")