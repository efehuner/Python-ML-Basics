# 1. IMPORT REQUIRED LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. LOAD DATASET
df = pd.read_csv("Telco-Customer-Churn.csv")

# 3. DATA PREPROCESSING
# FIX 'TotalCharges' COLUMN (CONVERT BLANK STRINGS TO NaN, THEN FILL WITH 0)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# DROP IRRELEVANT ID COLUMN
df = df.drop(columns=["customerID"])

# ENCODE CATEGORICAL VARIABLES (ONE-HOT ENCODING)
categorical_columns = df.select_dtypes(include=["object", "str"]).columns
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)

# 4. SPLIT DATA INTO FEATURES (X) AND TARGET (y)
X = df.drop(columns=["Churn_Yes"])
y = df["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# 5. TRAIN AND EVALUATE RANDOM FOREST
model_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)

print("\t--- RANDOM FOREST ---")
# FIXED: y_true COMES FIRST IN confusion_matrix
print(f"Confusion Matrix:\n{confusion_matrix(y_test, pred_rf)}")
print(f"Classification Report:\n{classification_report(y_test, pred_rf)}")
print(f"Accuracy: {accuracy_score(y_test, pred_rf):.4f}\n")

# 6. TRAIN AND EVALUATE XGBOOST
model_xgb = XGBClassifier(eval_metric="logloss", random_state=42)
model_xgb.fit(X_train, y_train)
pred_xgb = model_xgb.predict(X_test)

print("\t--- XGBOOST ---")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, pred_xgb)}")
print(f"Classification Report:\n{classification_report(y_test, pred_xgb)}")
print(f"Accuracy: {accuracy_score(y_test, pred_xgb):.4f}\n")

# 7. TRAIN AND EVALUATE LIGHTGBM
model_lgbm = LGBMClassifier(random_state=42, verbose=-1)
model_lgbm.fit(X_train, y_train)
pred_lgbm = model_lgbm.predict(X_test)

print("\t--- LIGHTGBM ---")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, pred_lgbm)}")
print(f"Classification Report:\n{classification_report(y_test, pred_lgbm)}")
print(f"Accuracy: {accuracy_score(y_test, pred_lgbm):.4f}\n")