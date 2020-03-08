import torch.nn as nn
import torch.nn.functional as F
import torch

from models.util import extract_args

class LSTMModel(nn.Module):
    def __init__(self, num_classes, input_dim=187, hidden_dim=100, layer_dim=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, num_classes)

        if 1 == num_classes:
            # compatible with nn.BCELoss
            self.softmax = nn.Sigmoid()
        else:
            # compatible with nn.CrossEntropyLoss
            self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        h_0 = torch.zeros(self.layer_dim, x.size(
            0), self.hidden_dim).requires_grad_()
        c_0 = torch.zeros(self.layer_dim, x.size(
            0), self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h_0.detach(), c_0.detach()))

        out = self.fc(out[:, -1, :])

        out = self.softmax(out)

        return out


def lstm_x(**kwargs):
    model = LSTMModel(**kwargs)
    return model


def lstm(model_name, num_classes, **kwargs):
    lstm_args = extract_args(kwargs, ["input_dim", "hidden_dim", "layer_dim"])

    return {
        'lstm': lstm_x(num_classes=num_classes, **lstm_args),
    }[model_name]
