import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv('creditcard.csv')

# Basic stats
print(data.head())
print(data.info())
print(data.describe())

# Check class distribution
print(data['Class'].value_counts())

from sklearn.utils import resample

# Separate majority and minority classes
majority = data[data['Class'] == 0]
minority = data[data['Class'] == 1]

# Upsample the minority class
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)

# Combine majority and upsampled minority class
balanced_data = pd.concat([majority, minority_upsampled])

print(balanced_data['Class'].value_counts())

# Split data into features and target
X = balanced_data.drop(['Class'], axis=1)
y = balanced_data['Class']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.show()

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load('fraud_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(data['features'])])
    return jsonify({'fraud': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)

import joblib
joblib.dump(model, 'fraud_model.pkl')

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('creditcard.csv')

# Sidebar
st.sidebar.title('Fraud Detection Dashboard')
st.sidebar.text('Explore fraud trends and patterns.')

# Main content
st.title('Credit Card Fraud Detection Insights')

if st.checkbox('Show raw data'):
    st.write(data.head())

# Visualizations
st.subheader('Transaction Class Distribution')
st.bar_chart(data['Class'].value_counts())




