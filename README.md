# Fraud Detection System with PyTorch

A fraud detection system that uses a neural network to classify credit card transactions as either fraudulent or legitimate.

## Tech Stack:
- Python: PyTorch for building the ML model
- Data: Credit Card Transactions Dataset (from Kaggle)
- Deployment: Flask API for real-time predictions
- Kubernetes: Deploy the model in a scalable microservice architecture

The Kaggle dataset is linked [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Steps/My Approach:
1. Data Preprocessing: Prepare the dataset, clean, and normalize it.
2. Build Model: Create a neural network in PyTorch for binary classification.
3. API: Use Flask to expose an API for fraud prediction.
4. Kubernetes: Containerize and deploy using Kubernetes.
