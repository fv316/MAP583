import torch.nn as nn
import torch.nn.functional as F
import torch


class LSTMModel(nn.Module):
    def __init__(self, num_classes, input_dim=187, hidden_dim=100, layer_dim=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # Building your LSTM
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
        # Initialize hidden state with zeros
        h_0 = torch.zeros(self.layer_dim, x.size(
            0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c_0 = torch.zeros(self.layer_dim, x.size(
            0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h_0.detach(), c_0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        out = self.softmax(out)
        return out


def lstm_1(**kwargs):
    model = LSTMModel(**kwargs)
    return model


def lstm(model_name, num_classes, pretrained=False):
    return{
        'lstm_1': lstm_1(num_classes=num_classes),
    }[model_name]
