import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math


# 将输入的词汇表索引转换为一个指定维度的embedding
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model, padding_idx=1):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=padding_idx)  # 指定索引为1的字符为填充符号


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEmbedding, self).__init__()
        self.encoder = torch.zeros(max_len, d_model, device=device)
        self.encoder.requires_grad_ = False
        pos = torch.arange(0, max_len, device=device).float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        self.encoder[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoder[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        seq_len = x.size(1)
        return self.encoder[:seq_len, :]


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, dropout_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model).to(device)
        self.pos_embedding = PositionalEmbedding(d_model, max_len, device)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        token_embedding = self.token_embedding(x)
        pos_embedding = self.pos_embedding(x)
        return self.dropout(token_embedding + pos_embedding)


if __name__ == "__main__":
    device = torch.device('cuda')  # Specify the device
    random_torch = torch.randint(0, 100, (4, 6), device=device)
    embedding = TransformerEmbedding(vocab_size=100, d_model=200, max_len=20, dropout_prob=0.2, device=device)
    print(embedding(random_torch))
