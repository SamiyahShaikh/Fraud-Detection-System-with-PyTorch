import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network architecture for fraud detection
class FraudDetectionModel(nn.Module):
    def __init__(self):
        super(FraudDetectionModel, self).__init__()
        # First fully connected layer with 30 input features and 64 output neurons
        self.fc1 = nn.Linear(30, 64)
        # Second fully connected layer
        self.fc2 = nn.Linear(64, 32)
        # Final layer for binary output
        self.fc3 = nn.Linear(32, 1)
        # Sigmoid activation function to output probabilities
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass through the network
        x = torch.relu(self.fc1(x))  # ReLU activation for hidden layers
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer
        return self.sigmoid(x)  # Apply sigmoid to get probabilities

# Initialize model, loss function, and optimizer
model = FraudDetectionModel()
criterion = nn.BCELoss()  # Binary Cross Entropy loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
