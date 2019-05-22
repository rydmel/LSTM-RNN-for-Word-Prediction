import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, layer_count, dropout):
        super().__init__()
        self.input_size = input_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.layer_count = layer_count
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, emb_size)
        self.rnn = nn.LSTM(emb_size, hidden_size, layer_count, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedded)

        return hidden, cell
