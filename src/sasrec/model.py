import torch
import torch.nn as nn

class SASRec(nn.Module):
    def __init__(self, n_items, max_len=50, d_model=64, n_head=4, n_layer=2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=0) 
        self.pos_emb = nn.Embedding(max_len, d_model) 
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.out = nn.Linear(d_model, n_items + 1)

    def forward(self, item_seq):
        seq_len = item_seq.size(1)
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0)
        inputs = self.item_emb(item_seq) + self.pos_emb(positions)
        feature = self.transformer(inputs)
        last_item_feature = feature[:, -1, :]
        return self.out(last_item_feature)