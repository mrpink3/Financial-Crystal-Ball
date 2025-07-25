# lstm_classifier.py

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Dataset
class StockClassificationDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.X = df.drop(columns=['movement_label']).values.astype('float32')
        self.y = df['movement_label'].values.astype('int64')

        
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
class StockLSTMClassifier(nn.Module):
    def __init__(self, input_size=14, hidden_size=64, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.classifier(hn[-1])
    
def get_predictions(model, dataloader, device):
    model.eval()
    preds = []
    truths = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds.extend(outputs.cpu().numpy().flatten())
            truths.extend(y.cpu().numpy().flatten())

    print(f"Length preds: {len(preds)}")
    print(f"Length truths: {len(truths)}")

    return np.array(preds), np.array(truths)


# Training + Evaluation
def evaluate_classification(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    return acc, f1, prec, rec, cm

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

        acc, f1, prec, rec, cm = evaluate_classification(model, dataloader, device)
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")
        print(f"Precision: {prec:.4f}, Recall: {rec:.4f}")
        print("Confusion Matrix:\n", cm)

# Main
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = StockClassificationDataset("LSTM Models/data/classification_data_returns.csv")

    X_train_raw, y_train = dataset.get_Train()
    X_test_raw, y_test = dataset.get_Test()
    #dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    X_train_returns = X_train_raw[:, :10]
    X_train_sentiments = X_train_raw[:, 10:]

    X_test_returns = X_test_raw[:, :10]
    X_test_sentiments = X_test_raw[:, 10:]

    X_train_flat_return = X_train_raw.reshape(X_train_returns.shape[0], -1)  # (num_samples, 14)
    X_test_flat_return = X_test_raw.reshape(X_test_returns.shape[0], -1)

    scaler = StandardScaler()
    X_train_returns_scaled = scaler.fit_transform(X_train_flat_return)
    X_test_returns_scaled = scaler.transform(X_test_flat_return)

    X_train_returns_scaled_r = X_train_returns_scaled.reshape(-1, 1, 14)
    X_test_returns_scaled_r = X_test_returns_scaled.reshape(-1, 1, 14)

    X_train = np.concatenate([X_train_returns_scaled_r, X_train_sentiments], axis=1)
    X_test = np.concatenate([X_test_returns_scaled_r, X_test_sentiments], axis=1)

    # Reshape back to 3D: (num_samples, 1, 14)
    # X_train = X_train.reshape(-1, 1, 14)
    # X_test = X_test.reshape(-1, 1, 14)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = StockLSTMClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    #train(model, dataloader, criterion, optimizer, device, epochs=1000)

    for epoch in range(100):
        train(model, train_loader, criterion, optimizer, device, epochs=1)
        acc, f1, prec, rec, cm = evaluate_classification(model, test_loader, device)
        print(f"[Test] Epoch {epoch+1}, Acc: {acc:.5f}, F1: {f1:.5f}, Precision: {prec:.5f}, Recall: {rec:.5f}")
        print("Confusion Matrix:\n", cm)

    #train(model, train_loader, criterion, optimizer, device, epochs=500)

    preds, truths = get_predictions(model, test_loader, device)

    df_results = pd.DataFrame({
        "True": truths,
        "Predicted": preds
    })

    df_results.to_csv("test_predictions_Classification.csv", index=False)

if __name__ == "__main__":
    main()
