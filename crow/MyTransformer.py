import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import embedding
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.init as init


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model, **kwargs):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, **kwargs)
        self.d_model = d_model

    def forward(self, x):
        return super().forward(x) * np.sqrt(d_model)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

    def embeddinglayer(self):
        return self.pe


def _generate_square_subsequent_mask(sz, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    # return torch.triu(torch.ones(sz, sz, device=device, dtype=dtype), diagonal=1) #
    # code from pytorch code
    return torch.triu(torch.full((sz, sz), float("-inf"), dtype=dtype, device=device), diagonal=1)


def subsequent_mask(sz):
    attn_shape = (1, sz, sz)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(1 - subsequent_mask)


if __name__ == '__main__':
    # pytorch_ code from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    vocab_size = 1000
    d_model = 512
    seed = 42
    max_len = 5000
    device = torch.device('cuda')  # Specify the device
    x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]).transpose(0, 1).to(torch.device('cuda'))
    print(x.size())

    # test 1
    pytorch_input_emb = nn.Embedding(vocab_size, d_model)
    init.normal_(pytorch_input_emb.weight, mean=0.0, std=1.0, generator=torch.manual_seed(seed))
    pytorch_input_emb.to(device)
    pytorch_x = pytorch_input_emb(x) * np.sqrt(d_model)
    print(pytorch_x.size())

    my_input_emb = TokenEmbedding(vocab_size, d_model)
    init.normal_(my_input_emb.weight, mean=0.0, std=1.0, generator=torch.manual_seed(seed))
    my_input_emb.to(device)
    my_x = my_input_emb(x)

    assert torch.equal(pytorch_x, my_x)

    # test 2
    my_pos_emb = PositionalEmbedding(d_model, dropout=0.1, max_len=max_len).to(device)
    my_x = my_pos_emb(my_x)

    # emb_layer = my_pos_emb.embeddinglayer().squeeze(1).to(torch.device('cpu'))
    # plt.figure(figsize=(15, 5))
    # pe = PositionalEmbedding(20, 0)
    # y = pe(torch.zeros(100, 1, 20))
    # plt.plot(np.arange(100), y[:, 0, 4:8].data.numpy())
    # plt.legend("dim %d" % p for p in [4, 5, 6, 7])
    # plt.show()

    # test 3
    mask = _generate_square_subsequent_mask(my_x.size(0))
    sm = subsequent_mask(my_x.size(0))
    print(mask.size())
    plt.figure(1, figsize=(10, 10))
    plt.imshow(mask.numpy())
    plt.figure(2, figsize=(10, 10))
    plt.imshow(sm[0].numpy())
    plt.show()
