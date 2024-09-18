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
        super(TSNN, self).__init__()
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


class TSCNNRE(nn.Module):
    def __init__(
        self,
        ts_input_size,
        meta_input_size,
        hidden_merge_size,
        hidden_ts_size,
        hidden_meta_size,
        output_size,
        nr_groups,
        num_random_effects,
        num_channels,
    ):
        super(TSCNNRE, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(ts_input_size, num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(num_channels, hidden_ts_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),  # This will ensure the output is of fixed size
        )
        self.fc_meta = nn.Linear(meta_input_size, hidden_meta_size)
        self.fc_merge = nn.Linear(hidden_ts_size + hidden_meta_size, hidden_merge_size)
        self.fc_output = nn.Linear(hidden_merge_size, output_size)
        self.dropout = nn.Dropout(0.5)
        self.random_effects = RandomEffectLayer(nr_groups, num_random_effects)

    def forward(self, ts, x, Z):
        ts = ts.transpose(
            1, 2
        )  # Reshape from [batch, seq, features] to [batch, features, seq]
        out_cnn = self.cnn(ts)
        out_cnn = out_cnn.squeeze(-1)  # Remove the last dimension after AdaptiveMaxPool
        meta_out = F.relu(self.fc_meta(x))
        combined = torch.cat((out_cnn, meta_out), dim=1)
        merged = F.relu(self.fc_merge(combined))
        merged = self.dropout(merged)
        output = self.fc_output(merged)
        return output + self.random_effects(Z)


class TSCNN(nn.Module):
    def __init__(
        self,
        ts_input_size,
        meta_input_size,
        hidden_merge_size,
        hidden_ts_size,
        hidden_meta_size,
        output_size,
        num_channels,
    ):
        super(TSCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(ts_input_size, num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(num_channels, hidden_ts_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),  # This will ensure the output is of fixed size
        )
        self.fc_meta = nn.Linear(meta_input_size, hidden_meta_size)
        self.fc_merge = nn.Linear(hidden_ts_size + hidden_meta_size, hidden_merge_size)
        self.fc_output = nn.Linear(hidden_merge_size, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, ts, x):
        ts = ts.transpose(
            1, 2
        )  # Reshape from [batch, seq, features] to [batch, features, seq]
        out_cnn = self.cnn(ts)
        out_cnn = out_cnn.squeeze(-1)  # Remove the last dimension after AdaptiveMaxPool
        meta_out = F.relu(self.fc_meta(x))
        combined = torch.cat((out_cnn, meta_out), dim=1)
        merged = F.relu(self.fc_merge(combined))
        merged = self.dropout(merged)
        output = self.fc_output(merged)
        return output
