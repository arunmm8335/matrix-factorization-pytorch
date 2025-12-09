import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import sys

# Add root to path so we can import if needed, though we use local imports here
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SASRec
from dataset import SASRecDataset

BATCH_SIZE = 64
EPOCHS = 20 # Increased to 20 to try and force learning order
LR = 0.001
MAX_LEN = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    if not os.path.exists('../../ml-latest-small'):
        os.system("wget -q https://files.grouplens.org/datasets/movielens/ml-latest-small.zip")
        os.system("unzip -q ml-latest-small.zip -d ../../")

    df = pd.read_csv('../../ml-latest-small/ratings.csv')
    item_ids = df["movieId"].unique().tolist()
    item2idx = {x: i+1 for i, x in enumerate(item_ids)}
    df["item"] = df["movieId"].map(item2idx)
    
    # Fix User Map
    user_ids = df["userId"].unique().tolist()
    user2idx = {x: i for i, x in enumerate(user_ids)}
    df["user"] = df["userId"].map(user2idx)
    
    num_items = len(item_ids)
    all_movie_ids = list(item2idx.values())

    dataset = SASRecDataset(df, all_movie_ids, max_len=MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SASRec(num_items, max_len=MAX_LEN).to(DEVICE)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Training SASRec on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for seq, pos, neg in loader:
            seq, pos = seq.to(DEVICE), pos.to(DEVICE)
            optimizer.zero_grad()
            logits = model(seq) 
            loss = criterion(logits, pos)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")

    # CRITIQUE
    model.eval()
    seq_A = torch.tensor([[1, 2] + [0]*(MAX_LEN-2)]).to(DEVICE) 
    seq_B = torch.tensor([[2, 1] + [0]*(MAX_LEN-2)]).to(DEVICE) 
    with torch.no_grad():
        pred_A = model(seq_A).argmax(dim=1).item()
        pred_B = model(seq_B).argmax(dim=1).item()
    print(f"\nOrder Check: [1,2]->{pred_A}, [2,1]->{pred_B}")

if __name__ == "__main__":
    main()