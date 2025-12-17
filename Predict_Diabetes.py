"""
Diabetes Prediction ML Project
Author: Pratima Dhende
Objective: Predict whether a person has diabetes using the Pima Indians Diabetes dataset.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# 1. Load Dataset Safely
# -------------------------------
csv_file = "diabetes.csv"
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, csv_file)

print("Loading data from:", file_path)
df = pd.read_csv(file_path)
print("Dataset loaded. Shape:", df.shape)
print(df.head())

# -------------------------------
# 2. Basic Data Cleaning / EDA (if needed)
# -------------------------------
print("Missing values:\n", df.isnull().sum())

# In this dataset zeros in certain columns indicate missing or invalid values — optionally clean:
cols_zero_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_zero_as_missing] = df[cols_zero_as_missing].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

# -------------------------------
# 3. Define Features and Target
# -------------------------------
X = df.drop('Outcome', axis=1)
y = df['Outcome']  # 0 = no diabetes, 1 = diabetes

# -------------------------------
# 4. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# -------------------------------
# 5. Train Random Forest Classifier
# -------------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# -------------------------------
# 6. Evaluate Model
# -------------------------------
acc = accuracy_score(y_test, y_pred)
print(f"\nRandom Forest Accuracy: {acc:.4f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: Diabetes Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("Confusion_matrix.png",dpi=300,bbox_inches="tight")
plt.show()

# -------------------------------
# 7. Feature Importance
# -------------------------------
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.title("Feature Importance for Diabetes Prediction")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.savefig("feature_importance.png",dpi=300,bbox_inches="tight")
plt.show()

# Outcome count plot(Target Distribution)
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Outcome')
plt.title("Distribution of Diabetic vs Non-Diabetic Patients")
plt.savefig("Outcome count plot(Target Distribution).png",dpi=300,bbox_inches="tight")
plt.show()

# Histogram / distribution Plot
plt.figure(figsize=(6,4))
sns.histplot(data=df, x='Glucose', hue='Outcome', kde=True)
plt.title("Glucose Distribution by Outcome")
plt.savefig("distribution_plot.png",dpi=300,bbox_inches="tight")
plt.show()

# Box Plot(Feature vs Outcome)
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Outcome', y='Insulin')
plt.title("Insulin Levels by Outcome")
plt.savefig("Feature vs outcome.png",dpi=300,bbox_inches="tight")
plt.show()

# Pairplot
sns.pairplot(df, hue='Outcome')
plt.subtitle("Pairplot of Diabetes Dataset")
plt.savefig("Pairplot(Outcome).png",dpi=300,bbox_inches="tight")
plt.show()

