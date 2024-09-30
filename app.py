from flask import Flask, request, jsonify
import torch
import numpy as np
from model import FraudDetectionModel

app = Flask(__name__)

# Load the trained fraud detection model
model = FraudDetectionModel()
model.load_state_dict(torch.load('fraud_model.pth'))   # Load saved model weights

@app.route('/predict', methods=['POST'])   # Define a route for predictions
def predict():
    data = request.json['transaction_data']   # Get the transaction data from the request
    data = np.array(data).reshape(1, -1)   # Reshape data to match model input
    data = torch.tensor(data, dtype=torch.float32)   # Convert to tensor
    prediction = model(data)   # Get model prediction
    result = "Fraud" if prediction.item() > 0.5 else "Legitimate"   # Determine if the transaction is fraudulent
    return jsonify({'prediction': result})   # Return the prediction as JSON

if __name__ == '__main__':
    app.run(debug=True)
