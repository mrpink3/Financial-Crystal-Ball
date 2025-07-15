# lstm_classifier.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Dataset
class StockClassificationDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.X = df.drop(columns=['movement_label']).values.astype('float32')
        self.y = df['movement_label'].values.astype('int64')
        self.X = torch.tensor(self.X).unsqueeze(1)
        self.y = torch.tensor(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Model
class StockLSTMClassifier(nn.Module):
    def __init__(self, input_size=11, hidden_size=64, num_classes=3):
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
    dataset = StockClassificationDataset("LSTM Models/data/classification_data.csv")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = StockLSTMClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model, dataloader, criterion, optimizer, device, epochs=20)

if __name__ == "__main__":
    main()
