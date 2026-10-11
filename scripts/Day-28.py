# 1. IMPORT REQUIRED LIBRARIES
from lightgbm import LGBMClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

# 2. LOAD DATA AND FIX COLUMN NAMES
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)

# RENAME COLUMNS TO REMOVE WHITESPACES
X.columns = X.columns.str.replace(" ", "_")
y = data.target

# 3. SPLIT THE DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# 4. INITIALIZE AND TRAIN LIGHTGBM MODEL (SILENT MODE)
model_lgbm = LGBMClassifier(random_state=42, verbose=-1)
model_lgbm.fit(X_train, y_train)

# 5. MAKE PREDICTIONS AND EVALUATE
predictions = model_lgbm.predict(X_test)
print(f"Success of LGBM: {accuracy_score(y_test, predictions):.4f}")

# 6. CALCULATE AND EXTRACT TOP 10 FEATURE IMPORTANCES
feature_importances = pd.Series(model_lgbm.feature_importances_, index=X.columns)
feature_importances_sorted = feature_importances.sort_values(ascending=False).head(10)
print("\nTop 10 Features:\n", feature_importances_sorted)

# 7. VISUALIZE WITH SEABORN
plt.figure(figsize=(12, 8))

# SET EXPLICIT X AND Y AXES FOR HORIZONTAL BAR PLOT
sns.barplot(x=feature_importances_sorted, y=feature_importances_sorted.index, palette="viridis")

plt.title("LGBM - Most Important 10 Features")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()