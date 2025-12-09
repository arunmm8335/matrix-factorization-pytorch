import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys

# Import local modules
from model import NeuralCollaborativeFiltering
from dataset import NCFDataset

# --- CONFIG ---
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. SHARED DATA LOADING
    # We look for data in the root /data folder or download it
    if not os.path.exists('../../ml-latest-small'):
        print("Downloading Data...")
        os.system("wget -q https://files.grouplens.org/datasets/movielens/ml-latest-small.zip")
        os.system("unzip -q ml-latest-small.zip -d ../../")

    # Load from the shared location
    df = pd.read_csv('../../ml-latest-small/ratings.csv')
    
    # Mapping
    user_ids = df["userId"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    item_ids = df["movieId"].unique().tolist()
    item2item_encoded = {x: i for i, x in enumerate(item_ids)}
    df["user"] = df["userId"].map(user2user_encoded)
    df["item"] = df["movieId"].map(item2item_encoded)

    train_df, _ = train_test_split(df, test_size=0.2, random_state=42)
    all_movies = list(item2item_encoded.values())

    dataset = NCFDataset(train_df, all_movies)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. MODEL SETUP
    model = NeuralCollaborativeFiltering(len(user2user_encoded), len(item2item_encoded)).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 3. TRAINING
    print(f"Training NCF on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for user, items, labels in loader:
            user = user.repeat_interleave(5).to(DEVICE)
            items = items.view(-1).to(DEVICE)
            labels = labels.view(-1).float().to(DEVICE)
            
            optimizer.zero_grad()
            output = model(user, items)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")

    # 4. CRITIQUE (Check Correlation)
    model.eval()
    user_a = torch.tensor([0] * 50).to(DEVICE)
    user_b = torch.tensor([100] * 50).to(DEVICE)
    items = torch.arange(0, 50).to(DEVICE)
    with torch.no_grad():
        preds_a = model(user_a, items).cpu().numpy()
        preds_b = model(user_b, items).cpu().numpy()
    
    corr = np.corrcoef(preds_a, preds_b)[0,1]
    print(f"\nFinal Correlation between User A and B: {corr:.4f}")

if __name__ == "__main__":
    main()