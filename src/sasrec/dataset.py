import torch
from torch.utils.data import Dataset
import random

class SASRecDataset(Dataset):
    def __init__(self, df, all_movie_ids, max_len=50):
        self.all_movie_ids = list(all_movie_ids)
        self.max_len = max_len
        df = df.sort_values(by=['user', 'timestamp'])
        self.user_sequences = df.groupby('user')['item'].apply(list).to_dict()
        self.users = list(self.user_sequences.keys())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        seq = self.user_sequences[user]
        seq_items = seq[:-1] 
        pos_item = seq[-1]   
        pad_len = self.max_len - len(seq_items)
        if pad_len > 0:
            seq_items = [0] * pad_len + seq_items
        else:
            seq_items = seq_items[-self.max_len:]
        neg_item = 0 
        return torch.tensor(seq_items), torch.tensor(pos_item), torch.tensor(neg_item)