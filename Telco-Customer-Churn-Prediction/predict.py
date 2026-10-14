# 1. IMPORT REQUIRED LIBRARIES
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 2. LOAD AND PREPROCESS DATA (SIMULATING NEW INCOMING DATA)
df = pd.read_csv("Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df = df.drop(columns=["customerID"])

categorical_columns = df.select_dtypes(include=["object", "str"]).columns
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)

# SEPARATE FEATURES AND TARGET
X = df.drop(columns=["Churn_Yes"])
y = df["Churn_Yes"]

# WE ONLY NEED THE TEST SET FOR EVALUATION HERE
_, X_test, _, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# 3. LOAD THE PRE-TRAINED MODEL FROM DISK
print("Loading pre-trained LightGBM model...")
loaded_model = joblib.load("lgbm_model.joblib")

# 4. MAKE PREDICTIONS ON THE TEST DATA
print("Making predictions...")
predictions = loaded_model.predict(X_test)

# 5. EVALUATE AND PRINT THE RESULTS
print(f"Loaded Model Accuracy: {accuracy_score(y_test, predictions):.4f}")