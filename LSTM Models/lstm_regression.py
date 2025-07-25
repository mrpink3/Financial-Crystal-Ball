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

        split_index = int(len(self.X) * 0.8)
        self.X_train, self.X_test = self.X[:split_index], self.X[split_index:]
        self.y_train, self.y_test = self.y[:split_index], self.y[split_index:]

        self.X_train = torch.tensor(self.X_train).unsqueeze(1)
        self.y_train = torch.tensor(self.y_train)
        self.X_test = torch.tensor(self.X_test).unsqueeze(1)
        self.y_test = torch.tensor(self.y_test)

        #self.X = torch.tensor(self.X).unsqueeze(1)
        #self.y = torch.tensor(self.y)

    def get_Train(self):
        return self.X_train, self.y_train
    
    def get_Test(self):
        return self.X_test, self.y_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Model
class StockLSTMRegressor(nn.Module):
    def __init__(self, input_size=14, hidden_size=64):
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
    
def get_predictions(model, dataloader, device):
    model.eval()
    preds = []
    truths = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds.extend(outputs.cpu().numpy())
            truths.extend(y.cpu().numpy())
    return np.array(preds), np.array(truths)

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

        #mae, rmse, r2 = evaluate_regression(model, dataloader, device)
        #print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, MAE: {mae:.5f}, RMSE: {rmse:.5f}, R²: {r2:.4f}")



# Main
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = StockRegressionDataset("LSTM Models/data/regression_data.csv")

    X_train, y_train = dataset.get_Train()
    X_test, y_test = dataset.get_Test()
    #dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = StockLSTMRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        train(model, train_loader, criterion, optimizer, device, epochs=1)
        mae, rmse, r2 = evaluate_regression(model, test_loader, device)
        print(f"[Test] Epoch {epoch+1}, MAE: {mae:.5f}, RMSE: {rmse:.5f}, R²: {r2:.4f}")

    preds, truths = get_predictions(model, test_loader, device)

    df_results = pd.DataFrame({
        "True": truths,
        "Predicted": preds
    })

    df_results.to_csv("test_predictions_Regression.csv", index=False)

if __name__ == "__main__":
    main()
