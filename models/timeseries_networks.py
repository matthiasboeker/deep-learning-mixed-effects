import torch
import torch.nn as nn
from models.models import RandomEffectLayer
import torch.nn.functional as F


class TSNNRE(nn.Module):
    def __init__(
        self,
        ts_input_size,
        meta_input_size,
        hidden_merge_size,
        hidden_ts_size,
        hidden_meta_size,
        num_layers,
        output_size,
        nr_groups,
        num_random_effects,
    ):
        super(TSNNRE, self).__init__()
        self.rnn = nn.GRU(ts_input_size, hidden_ts_size, num_layers, batch_first=True)
        self.fc_meta = nn.Linear(meta_input_size, hidden_meta_size)
        self.fc_merge = nn.Linear(hidden_ts_size + hidden_meta_size, hidden_merge_size)
        self.fc_output = nn.Linear(hidden_merge_size, output_size)
        self.dropout = nn.Dropout(0.5)
        self.random_effects = RandomEffectLayer(nr_groups, num_random_effects)
        self.num_layers = num_layers
        self.hidden_ts_size = hidden_ts_size

    def forward(self, ts, x, Z):
        h0 = torch.zeros(self.num_layers, ts.size(0), self.hidden_ts_size).to(ts.device)
        out_rnn, _ = self.rnn(ts, h0)  # [batch, sequence, rnn_hiddensize]
        out_rnn = out_rnn[:, -1, :]  # [batch, rnn_hiddensize]
        meta_out = F.relu(self.fc_meta(x))  # [batch, meta_hiddensize]
        combined = torch.cat((out_rnn, meta_out), dim=1)
        merged = F.relu(self.fc_merge(combined))
        merged = self.dropout(merged)
        merged = self.fc_output(merged)
        return merged + self.random_effects(Z)


class TSNN(nn.Module):
    def __init__(
        self,
        ts_input_size,
        meta_input_size,
        hidden_merge_size,
        hidden_ts_size,
        hidden_meta_size,
        num_layers,
        output_size,
    ):
        super(TSNN, self).__init__()
        self.rnn = nn.GRU(ts_input_size, hidden_ts_size, num_layers, batch_first=True)
        self.fc_meta = nn.Linear(meta_input_size, hidden_meta_size)
        self.fc_merge = nn.Linear(hidden_ts_size + hidden_meta_size, hidden_merge_size)
        self.fc_output = nn.Linear(hidden_merge_size, output_size)
        self.dropout = nn.Dropout(0.5)
        self.num_layers = num_layers
        self.hidden_ts_size = hidden_ts_size

    def forward(self, ts, x):
        h0 = torch.zeros(self.num_layers, ts.size(0), self.hidden_ts_size).to(ts.device)
        out_rnn, _ = self.rnn(ts, h0)  # [batch, sequence, rnn_hidden_size]
        out_rnn = out_rnn[:, -1, :]  # [batch, rnn_hidden_size]
        meta_out = F.relu(self.fc_meta(x))  # [batch, meta_hidden_size]
        combined = torch.cat((out_rnn, meta_out), dim=1)
        merged = F.relu(self.fc_merge(combined))
        merged = self.dropout(merged)
        return self.fc_output(merged)
