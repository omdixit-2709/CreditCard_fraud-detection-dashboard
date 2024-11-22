# Credit Card Fraud Detection Dashboard

This project is a **Credit Card Fraud Detection** system built with machine learning, deployed as an interactive dashboard using **Streamlit**. The goal of the project is to predict whether a credit card transaction is fraudulent or not using various machine learning models. The dashboard provides an interactive user interface to explore the dataset and the model's performance.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Technology Stack](#technology-stack)
- [Features](#features)
- [How to Run the Project](#how-to-run-the-project)
- [Results & Evaluation](#results-evaluation)
- [License](#license)

## Overview
The **Credit Card Fraud Detection Dashboard** allows users to explore the dataset, visualize trends, and test the fraud detection model's performance. It uses machine learning techniques, including **Random Forest Classifier**, to detect fraudulent transactions based on historical data.

- **Objective**: Classify whether a credit card transaction is fraudulent or not.
- **Dataset**: A publicly available dataset of credit card transactions labeled as fraudulent or legitimate.
- **Frameworks**: The project uses **Streamlit** for the frontend dashboard and **scikit-learn** for machine learning.

## Dataset
The dataset used in this project contains anonymized features of credit card transactions. It includes both **fraudulent** and **non-fraudulent** transactions. This data is used to train and evaluate a machine learning model for detecting fraud.

- **Size**: 143 MB
- **Features**: The dataset consists of various transaction details such as time, amount, and other anonymized features.
- **Class Label**: The `Class` column contains the target label:
  - `0`: Legitimate transaction
  - `1`: Fraudulent transaction

> **Note**: The dataset is too large to be uploaded directly to GitHub, and can be accessed via [this release](https://github.com/omdixit-2709/CreditCard_fraud-detection-dashboard/releases/tag/v1.0).

## Technology Stack
- **Backend**: Python, scikit-learn (for machine learning models)
- **Frontend**: Streamlit (for interactive dashboard)
- **Libraries**: 
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - Flask (for API deployment)

## Features
- **Explore the Dataset**: View the raw data, transaction distribution, and features of the credit card transactions.
- **Visualizations**: Interactive charts showing the class distribution, feature importances, and confusion matrix.
- **Model Evaluation**: View model performance metrics, such as precision, recall, and F1 score, through visualizations.
- **Fraud Prediction**: Use the trained model to predict whether a new transaction is fraudulent or not by submitting features through an API.

## How to Run the Project

### **1. Clone the Repository**

```bash
git clone https://github.com/omdixit-2709/CreditCard_fraud-detection-dashboard.git
cd CreditCard_fraud-detection-dashboard

###Install Dependencies
Ensure you have Python 3.x installed, then install the required packages:

pip install -r requirements.txt

###Run the Dashboard
To run the Streamlit dashboard locally, use the following command:

streamlit run app.py


### Explanation of Each Section:
1. **Overview**: Describes the purpose of the project and its goals.
2. **Dataset**: Provides details about the dataset, including the size, columns, and what each class label represents.
3. **Technology Stack**: Lists the frameworks and libraries used in the project.
4. **Features**: Describes the key functionalities and features of the project.
5. **How to Run the Project**: Gives instructions on how to clone, install dependencies, and run both the **Streamlit Dashboard** and the **Flask API**.
6. **Results & Evaluation**: Provides an overview of the model's performance.
7. **License**: Specifies the licensing terms for the repository (MIT License is a common choice for open-source projects).

### How to Add This `README.md` to Your Repository:
1. Create a new file in your project directory named `README.md`.
2. Paste the contents above into this file.
3. Add, commit, and push the changes:

```bash
git add README.md
git commit -m "Add README.md file"
git push origin main
