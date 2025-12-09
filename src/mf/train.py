import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from model import MatrixFactorizationWithBias
from dataset import MovieLensDataset

# --- CONFIG ---
BATCH_SIZE = 128
EPOCHS = 50
LR = 0.05
FACTORS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. DOWNLOAD & PREPARE DATA
    if not os.path.exists('ml-latest-small'):
        print("Downloading MovieLens data...")
        os.system("wget -q https://files.grouplens.org/datasets/movielens/ml-latest-small.zip")
        os.system("unzip -q ml-latest-small.zip")

    print("Loading data...")
    df = pd.read_csv('./ml-latest-small/ratings.csv')

    # Mapping IDs
    user_ids = df["userId"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    item_ids = df["movieId"].unique().tolist()
    item2item_encoded = {x: i for i, x in enumerate(item_ids)}

    df["user"] = df["userId"].map(user2user_encoded)
    df["item"] = df["movieId"].map(item2item_encoded)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_loader = DataLoader(
        MovieLensDataset(train_df["user"].values, train_df["item"].values, train_df["rating"].values),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        MovieLensDataset(test_df["user"].values, test_df["item"].values, test_df["rating"].values),
        batch_size=BATCH_SIZE
    )

    # 2. INIT MODEL
    global_mean = train_df["rating"].mean()
    model = MatrixFactorizationWithBias(
        len(user2user_encoded), len(item2item_encoded), global_mean, FACTORS
    ).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=1e-5)

    # 3. TRAINING LOOP
    print(f"Starting Training on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for users, items, ratings in train_loader:
            users, items, ratings = users.to(DEVICE), items.to(DEVICE), ratings.to(DEVICE)
            optimizer.zero_grad()
            output = model(users, items)
            loss = criterion(output, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}: Avg Loss = {total_loss/len(train_loader):.4f}")

    # 4. EVALUATION
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for users, items, ratings in test_loader:
            users, items, ratings = users.to(DEVICE), items.to(DEVICE), ratings.to(DEVICE)
            output = model(users, items)
            test_loss += criterion(output, ratings).item()
    
    rmse = np.sqrt(test_loss / len(test_loader))
    print(f"\nFinal Test RMSE: {rmse:.4f}")

    # 5. CRITIQUE (Popularity Bias Check)
    print("\n--- CRITIQUE: Popularity Bias Check ---")
    user_a = torch.tensor([0] * 50).to(DEVICE)
    user_b = torch.tensor([100] * 50).to(DEVICE) # Different user
    all_items = torch.arange(0, 50).to(DEVICE)
    
    preds_a = model(user_a, all_items).detach().cpu().numpy()
    preds_b = model(user_b, all_items).detach().cpu().numpy()
    
    correlation = np.corrcoef(preds_a, preds_b)[0,1]
    print(f"Correlation between predictions for User 0 and User 100: {correlation:.4f}")
    
if __name__ == "__main__":
    main()