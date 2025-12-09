import torch
from torch.utils.data import Dataset
import random

class NCFDataset(Dataset):
    def __init__(self, ratings_df, all_movie_ids):
        self.users = torch.tensor(ratings_df["user"].values)
        self.items = torch.tensor(ratings_df["item"].values)
        self.all_movie_ids = set(all_movie_ids)
        self.user_set = ratings_df.groupby("user")["item"].apply(set).to_dict()

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.items[idx]
        
        # Negative Sampling: Pick 4 movies user hasn't seen
        neg_items = []
        seen_items = self.user_set.get(user.item(), set())
        while len(neg_items) < 4:
            choice = random.choice(list(self.all_movie_ids))
            if choice not in seen_items and choice != pos_item:
                neg_items.append(choice)
                
        # Return format: User, [Pos, Neg...], [1, 0...]
        items = [pos_item] + neg_items
        labels = [1.0] + [0.0] * 4
        
        return user, torch.tensor(items), torch.tensor(labels)