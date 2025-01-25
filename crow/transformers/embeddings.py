class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000, device=torch.device('cpu'), **kwargs):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        embedding = torch.zeros(max_len, d_model, device=device)
        pos = torch.arange(0, max_len, device=device).float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        embedding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        embedding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        embedding = embedding.unsqueeze(0)
        self.register_buffer('embedding', embedding)

    def forward(self, x):
        x = x + Variable(self.embedding[:, x.size(1)], requires_grad=False)
        return self.dropout(x)

    def get_embedding(self):
        return self.embedding


class TransformerModel(nn.Transformer):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nhid,
                                               num_encoder_layers=nlayers)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz, sz)))

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)


if __name__ == '__main__':
    # compare with pytorch offical version
    # https://github.com/pytorch/examples/blob/main/word_language_model/model.py



    # word /token embedding
    # token_embedding = TokenEmbedding(vocab_size, d_model, device=device)
    # x = token_embedding(x)
    #
    # # position embedding
    # positioanl_embedding = PositionalEmbedding(d_model, 0.1, 60, device=device)
    # x = positioanl_embedding(x)
    #
    # # plot positioanl_embedding
    # positioanl_embedding = positioanl_embedding.get_embedding().to(torch.device('cpu'))
    # embedding_data = positioanl_embedding[0].numpy()
    #
    # plt.figure(figsize=(12, 8))
    # plt.imshow(embedding_data, aspect='auto', cmap='viridis')
    # plt.colorbar(label='Embedding value')
    # plt.title("Positional Embedding Visualization")
    # plt.xlabel("Embedding Dimension")
    # plt.ylabel("Position")
    # plt.show()
    # # input = torch.LongTensor([[1, 2, 4, 5], [1, 3, 2, 9]])
    # # print(input.shape)
    # #
    # # result = embedding(input)
    # # print(result)
    # # print(result.shape)
    # #
    # # embedding = TokenEmbedding(10, 3, padding_idx=0)
    # # input = torch.LongTensor([[1, 0, 4, 5], [1, 0, 2, 9]])
    # # result = embedding(input)
    # # print(result)
    # # print(result.shape)
