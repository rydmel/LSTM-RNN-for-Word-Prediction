import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMDecoder(nn.Module):
    def __init__(self, output_size, emb_size, hidden_size, layer_count, dropout):
        super().__init__()

        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layer_count = layer_count
        self.dropout = dropout
        self.embedding = nn.Embedding(output_size, emb_size)
        self.rnn = nn.LSTM(emb_size, hidden_size, layer_count, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):

        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        pred = self.out(output.squeeze(0))

        return pred, hidden, cell