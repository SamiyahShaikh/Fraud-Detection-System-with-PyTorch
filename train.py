import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Load the credit card fraud dataset
df = pd.read_csv('creditcard.csv')
X = df.iloc[:, :-1].values  # Features (all columns except the last one)
y = df.iloc[:, -1].values    # Labels (last column)

# Data Preprocessing
scaler = StandardScaler()  # Standardize features by removing the mean and scaling to unit variance
X_scaled = scaler.fit_transform(X)  # Fit and transform the feature data

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Create a DataLoader for the training set
train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                           torch.tensor(y_train, dtype=torch.float32))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)  # Shuffle the data for training

# Train the model
model = FraudDetectionModel()  # Initialize the model
epochs = 10  # Set the number of training epochs
for epoch in range(epochs):
    for data, target in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        output = model(data)  # Forward pass
        loss = criterion(output.squeeze(), target)  # Calculate the loss
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the model parameters

print("Training complete.")  # Notify when training is finished
