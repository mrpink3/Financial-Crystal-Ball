# lstm_regressor.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Dataset
class StockRegressionDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.X = df.drop(columns=['next_return']).values.astype('float32')
        self.y = df['next_return'].values.astype('float32')
        self.X = torch.tensor(self.X).unsqueeze(1)
        self.y = torch.tensor(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Model
class StockLSTMRegressor(nn.Module):
    def __init__(self, input_size=11, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.regressor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.regressor(hn[-1]).squeeze(-1)

# Evaluation
def evaluate_regression(model, dataloader, device):
    model.eval()
    preds = []
    truths = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds.extend(outputs.cpu().numpy())
            truths.extend(y.cpu().numpy())

    preds = np.array(preds)
    truths = np.array(truths)

    mae = mean_absolute_error(truths, preds)
    rmse = np.sqrt(mean_squared_error(truths, preds))
    r2 = r2_score(truths, preds)
    return mae, rmse, r2

# Train Loop
def train(model, dataloader, loss_fn, optimizer, device, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        mae, rmse, r2 = evaluate_regression(model, dataloader, device)
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, MAE: {mae:.5f}, RMSE: {rmse:.5f}, R²: {r2:.4f}")

# Main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = StockRegressionDataset("LSTM Models/data/regression_data.csv")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = StockLSTMRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model, dataloader, criterion, optimizer, device, epochs=20)

if __name__ == "__main__":
    main()
