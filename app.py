from flask import Flask, request, jsonify
import torch
import numpy as np
from model import FraudDetectionModel

app = Flask(__name__)

model = FraudDetectionModel()
model.load_state_dict(torch.load('fraud_model.pth'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['transaction_data']
    data = np.array(data).reshape(1, -1)
    data = torch.tensor(data, dtype=torch.float32)
    prediction = model(data)
    result = "Fraud" if prediction.item() > 0.5 else "Legitimate"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
