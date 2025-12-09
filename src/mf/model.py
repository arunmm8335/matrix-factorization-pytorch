import torch
import torch.nn as nn

class MatrixFactorizationWithBias(nn.Module):
    def __init__(self, n_users, n_items, global_mean, n_factors=20):
        super().__init__()
        # 1. Embeddings (The Vectors q and p)
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        
        # 2. Biases (The Terms b_u and b_i) - Cited from Eq 4 of Koren paper
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        
        # 3. Global Average (mu)
        self.global_mean = nn.Parameter(torch.tensor(global_mean))
        
        # Initialization (Important for convergence)
        self.user_factors.weight.data.uniform_(0, 0.05)
        self.item_factors.weight.data.uniform_(0, 0.05)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

    def forward(self, user, item):
        # Dot product part
        dot = (self.user_factors(user) * self.item_factors(item)).sum(1)
        
        # Bias part (Eq 4: mu + b_i + b_u + dot)
        bi = self.item_bias(item).squeeze()
        bu = self.user_bias(user).squeeze()
        
        return self.global_mean + bi + bu + dot