# 1. IMPORT REQUIRED LIBRARIES
import pandas as pd
import pickle
import joblib
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# 2. LOAD AND PREPROCESS DATA (FROM DAY 30)
df = pd.read_csv("Telco-Customer-Churn.csv")

# FIX 'TotalCharges' AND DROP 'customerID'
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)
df = df.drop(columns=["customerID"])

# ENCODE CATEGORICAL VARIABLES
categorical_columns = df.select_dtypes(include=["object", "str"]).columns
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)

# SPLIT DATA
X = df.drop(columns=["Churn_Yes"])
y = df["Churn_Yes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# 3. TRAIN THE CHAMPION MODEL (LIGHTGBM)
print("Training LightGBM model...")
model_lgbm = LGBMClassifier(random_state=42, verbose=-1)
model_lgbm.fit(X_train, y_train)

# CHECK ORIGINAL ACCURACY
original_pred = model_lgbm.predict(X_test)
print(f"Original LightGBM Accuracy: {accuracy_score(y_test, original_pred):.4f}\n")

print("--- SAVING MODELS ---")

# 4. SAVE MODEL USING PICKLE (wb = write binary)
with open("lgbm_model.pkl", "wb") as file:
    pickle.dump(model_lgbm, file)
print("Model successfully saved as 'lgbm_model.pkl'")

# 5. SAVE MODEL USING JOBLIB (Optimized for Machine Learning)
joblib.dump(model_lgbm, "lgbm_model.joblib")
print("Model successfully saved as 'lgbm_model.joblib'\n")

print("--- LOADING MODELS AND PREDICTING ---")

# 6. LOAD MODEL WITH PICKLE (rb = read binary)
with open("lgbm_model.pkl", "rb") as file:
    loaded_pickle_model = pickle.load(file)

# 7. LOAD MODEL WITH JOBLIB
loaded_joblib_model = joblib.load("lgbm_model.joblib")

# 8. PROVE THEY WORK (COMPARE ACCURACIES)
pickle_pred = loaded_pickle_model.predict(X_test)
joblib_pred = loaded_joblib_model.predict(X_test)

print(f"Loaded Pickle Model Accuracy: {accuracy_score(y_test, pickle_pred):.4f}")
print(f"Loaded Joblib Model Accuracy: {accuracy_score(y_test, joblib_pred):.4f}")