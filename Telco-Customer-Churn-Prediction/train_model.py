# 1. IMPORT REQUIRED LIBRARIES
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# 2. LOAD AND PREPROCESS DATA
df = pd.read_csv("Telco-Customer-Churn.csv")

# CONVERT TOTAL CHARGES TO NUMERIC AND FILL MISSING VALUES WITH 0
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

# DROP IRRELEVANT CUSTOMER ID COLUMN
df = df.drop(columns=["customerID"])

# ENCODE CATEGORICAL VARIABLES (ONE-HOT ENCODING)
categorical_columns = df.select_dtypes(include=["object", "str"]).columns
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)

# 3. SPLIT DATA INTO FEATURES (X) AND TARGET (y)
X = df.drop(columns=["Churn_Yes"])
y = df["Churn_Yes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# 4. INITIALIZE AND TRAIN THE LIGHTGBM MODEL
print("Training LightGBM model...")
model_lgbm = LGBMClassifier(random_state=42, verbose=-1)
model_lgbm.fit(X_train, y_train)

# EVALUATE MODEL PERFORMANCE
predictions = model_lgbm.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions):.4f}")

# 5. SAVE THE TRAINED MODEL TO DISK USING JOBLIB
joblib.dump(model_lgbm, "lgbm_model.joblib")
print("Model successfully saved as 'lgbm_model.joblib'")