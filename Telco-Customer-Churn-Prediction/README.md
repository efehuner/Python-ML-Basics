# 📉 Telco Customer Churn Prediction & Model Deployment

This project aims to predict customer churn (whether a customer will leave the telecommunications provider) using machine learning. The project demonstrates the end-to-end data science lifecycle: data preprocessing, model selection, and model serialization for deployment.

## 🚀 Project Overview

Customer retention is crucial for telecommunication companies. In this project, I processed the **Telco Customer Churn** dataset and compared multiple algorithms (Random Forest, XGBoost, and LightGBM). **LightGBM** achieved the best performance and was serialized using `Joblib` for future deployment in production environments.

## 🛠️ Tech Stack

- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, LightGBM, XGBoost
- **Model Serialization:** Joblib, Pickle

## 📂 Project Structure

- `train_model.py`: Script to load data, handle missing values (e.g., in `TotalCharges`), perform One-Hot Encoding, train the LightGBM model, and save it to disk.
- `predict.py`: A lightweight script simulating a production environment. It loads the pre-trained `lgbm_model.joblib` and makes predictions on new customer data.
- `lgbm_model.joblib`: The serialized LightGBM model ready for deployment.

## 📊 Quick Start

To test the pre-trained model on your local machine without retraining:

1. Clone the repository:
   ```bash
   git clone [https://github.com/KULLANICI_ADIN/Telco-Customer-Churn-Prediction.git](https://github.com/KULLANICI_ADIN/Telco-Customer-Churn-Prediction.git)
   cd Telco-Customer-Churn-Prediction
   ```
