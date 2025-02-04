import copy
import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from transformers.models.fsmt.modeling_fsmt import DecoderLayer


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def sequence_mask(size):
    ones = torch.from_numpy(np.ones((size, size))).float()
    return 1 - torch.triu(ones, diagonal=1)


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    x = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        x = x.masked_fill(mask == 0, -1e9)
    p_attn = torch.softmax(x, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        len_q = query.size(1)
        q = self.linears[0](query).view(batch_size, len_q, -1, self.d_k).transpose(1, 2)
        k = self.linears[1](key).view(batch_size, len_q, -1, self.d_k).transpose(1, 2)
        v = self.linears[2](value).view(batch_size, len_q, -1, self.d_k).transpose(1, 2)
        atten, _ = attention(q, k, v, mask, dropout=self.dropout)
        return self.linears[3](atten.transpose(1, 2).contiguous().view(batch_size, len_q, -1))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w2(self.dropout(self.relu(self.w1(x))))


class NormLayer(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(NormLayer, self).__init__()
        self.eps = eps
        self.a = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer):
        return self.dropout(sublayer(self.norm(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, pos_ffn, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.pos_ffn = pos_ffn
        self.sublayer0 = SublayerConnection(d_model, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)

    def forward(self, x, mask):
        x = self.sublayer0(x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer1(x, self.pos_ffn)


class TransformerEncoder(nn.Module):
    def __init__(self, layer, N):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = NormLayer(layer.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, pos_ffn, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.pos_ffn = pos_ffn
        self.sublayer0 = SublayerConnection(d_model, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x, memory, source_mask, target_mask):
        m = memory
        x = self.sublayer0(x, lambda x: self.self_attn(x, x, x, mask=target_mask))
        x = self.sublayer1(x, lambda x: self.src_attn(x, m, m, mask=source_mask))
        return self.sublayer1(x, self.pos_ffn)


class TransformerDecoder(nn.Module):
    def __init__(self, layer, N):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = NormLayer(layer.d_model)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), -1)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embed = source_embed
        self.target_embed = target_embed
        self.generator = generator

    def forward(self, source, source_mask, target, target_mask):
        return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)

    def encode(self, source, source_mask):
        return self.encoder(self.source_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        return self.decoder(self.target_embed(target), memory, source_mask, target_mask)


def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, num_head=8, dropout=0.1):
    c = copy.deepcopy
    atten = MultiHeadAttention(d_model, num_head, dropout)
    pos_ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

    pos_embed = PositionalEncoding(d_model=d_model, dropout=dropout)

    model = EncoderDecoder(
        encoder=TransformerEncoder(TransformerEncoderLayer(d_model, c(atten), c(pos_ffn), dropout), N),
        decoder=TransformerDecoder(TransformerDecoderLayer(d_model, c(atten), c(atten), c(pos_ffn), dropout), N),
        source_embed=nn.Sequential(nn.Embedding(source_vocab, d_model), c(pos_embed)),
        target_embed=nn.Sequential(nn.Embedding(source_vocab, d_model), c(pos_embed)),
        generator=Generator(d_model, target_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


if __name__ == '__main__':
    vocab_size = 1000
    d_model = 512
    dropout = 0.2
    source = torch.tensor([[100, 2, 421, 508], [491, 998, 1, 221]])
    target = torch.tensor([[100, 2, 421, 508], [491, 998, 1, 221]])

    source_embed = nn.Embedding(vocab_size, d_model)
    target_embed = nn.Embedding(vocab_size, d_model)

    self_attn = MultiHeadAttention(d_model=d_model, num_heads=8, dropout=dropout)
    pos_ffn = PositionwiseFeedForward(512, 64, dropout=dropout)
    ecl = TransformerEncoderLayer(d_model=d_model, self_attn=self_attn, pos_ffn=pos_ffn, dropout=dropout)
    encoder = TransformerEncoder(ecl, 5)

    self_attn = MultiHeadAttention(d_model=d_model, num_heads=8, dropout=dropout)
    src_attn = MultiHeadAttention(d_model=d_model, num_heads=8, dropout=dropout)
    pos_ffn = PositionwiseFeedForward(512, 64, dropout=dropout)
    dcl = TransformerDecoderLayer(d_model=d_model, self_attn=self_attn, src_attn=self_attn, pos_ffn=pos_ffn,
                                  dropout=dropout)
    decoder = TransformerDecoder(dcl, 5)

    gen = Generator(d_model=d_model, vocab_size=vocab_size)

    source_mask = torch.zeros((4, 4))
    target_mask = torch.zeros((4, 4))

    ed = EncoderDecoder(encoder, decoder, source_embed, target_embed, gen)
    ed_result = ed(source, source_mask, target, target_mask)
    print(ed_result)
    print(ed_result.shape)

    model = make_model(vocab_size, vocab_size, 6, d_model, 64, 8, 0.1)
    model_result = model(source, source_mask, target, target_mask)
    print(model_result)
    print(model_result.shape)
    print(model)
