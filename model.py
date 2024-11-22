# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from flask import Flask, request, jsonify
import joblib
import streamlit as st

# ====================== Data Preprocessing ====================== #
# Load dataset
try:
    data = pd.read_csv('creditcard.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: File 'creditcard.csv' not found. Please ensure the file is in the working directory.")
    exit()

# Basic information about the dataset
print("First few rows of the dataset:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

# Check for class imbalance
print("\nClass Distribution in the Dataset:")
print(data['Class'].value_counts())

# Handling Imbalanced Dataset
# Separate the majority and minority classes
majority = data[data['Class'] == 0]
minority = data[data['Class'] == 1]

# Upsample the minority class to balance the dataset
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)

# Combine the majority and upsampled minority classes
balanced_data = pd.concat([majority, minority_upsampled])

# Verify the new class distribution
print("\nBalanced Class Distribution:")
print(balanced_data['Class'].value_counts())

# ====================== Model Training and Evaluation ====================== #
# Split data into features (X) and target variable (y)
X = balanced_data.drop(['Class'], axis=1)
y = balanced_data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("\nRandom Forest model trained successfully!")

# Evaluate the model's performance
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature importance visualization
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.show()

# Save the trained model for deployment
joblib.dump(model, 'fraud_model.pkl')
print("Model saved as 'fraud_model.pkl'")

# ====================== Flask API for Predictions ====================== #
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict fraud.
    Accepts JSON data with 'features' key, containing a list of feature values.
    Returns whether the transaction is fraudulent.
    """
    data = request.get_json(force=True)
    prediction = model.predict([np.array(data['features'])])
    return jsonify({'fraud': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)

# ====================== Streamlit Dashboard ====================== #
# Load dataset for the dashboard
try:
    data = pd.read_csv('creditcard.csv')
except FileNotFoundError:
    st.error("Dataset 'creditcard.csv' not found. Please upload the file to proceed.")

# Sidebar for navigation
st.sidebar.title('Fraud Detection Dashboard')
st.sidebar.text('Explore fraud trends and patterns.')

# Main dashboard content
st.title('Credit Card Fraud Detection Insights')

if st.checkbox('Show raw data'):
    st.write(data.head())

# Visualization of class distribution
st.subheader('Transaction Class Distribution')
st.bar_chart(data['Class'].value_counts())
