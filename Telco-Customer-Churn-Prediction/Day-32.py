# 1. IMPORT REQUIRED LIBRARIES
import streamlit as st
import joblib
import pandas as pd

# 2. LOAD THE SAVED MODEL
# Dün kaydettiğimiz LightGBM modelini yüklüyoruz
model = joblib.load('lgbm_model.joblib')

# 3. DESIGN THE USER INTERFACE
st.title("📞 Telco Customer Churn Predictor")
st.write("Please enter the core customer details below to predict churn risk:")

# GET USER INPUTS (Using the most important numerical features for simplicity)
tenure = st.slider("Tenure (Months with company)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=120.0, value=50.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0)

# 4. PREDICTION BUTTON LOGIC
if st.button("Predict Customer Churn"):
    
    # TRICK: To prevent column-mismatch errors, we quickly read the original CSV headers
    # and create a dataframe of zeros with the exact same columns our model expects.
    df_template = pd.read_csv("Telco-Customer-Churn.csv")
    df_template["TotalCharges"] = pd.to_numeric(df_template["TotalCharges"], errors="coerce").fillna(0)
    df_template = df_template.drop(columns=["customerID"])
    cat_cols = df_template.select_dtypes(include=["object", "str"]).columns
    df_template = pd.get_dummies(df_template, columns=cat_cols, drop_first=True, dtype=int)
    
    # Extract expected column names (excluding the target column)
    expected_columns = df_template.drop(columns=["Churn_Yes"]).columns
    
    # Create a 1-row DataFrame filled with 0s
    input_df = pd.DataFrame(0, index=[0], columns=expected_columns)
    
    # Insert the user's input into the correct columns
    input_df["tenure"] = tenure
    input_df["MonthlyCharges"] = monthly_charges
    input_df["TotalCharges"] = total_charges

    # 5. MAKE PREDICTION
    prediction = model.predict(input_df)

    # 6. DISPLAY RESULTS
    if prediction[0] == 1:
        st.error("⚠️ WARNING: High risk of churn! This customer might leave.")
    else:
        st.success("✅ SAFE: This customer is likely to stay.")