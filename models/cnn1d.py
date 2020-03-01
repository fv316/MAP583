import torch.nn as nn
import torch.nn.functional as F


class Cnn1d(nn.Module):
    def __init__(self, num_classes=5, input_channels=1):
        super(Cnn1d, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_channels,
                               out_channels=8, kernel_size=7, stride=1)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv1d(8, 32, kernel_size=7, stride=1)
        self.fc1 = nn.Linear(in_features=576, out_features=num_classes)

        if 1 == num_classes:
            # compatible with nn.BCELoss
            self.softmax = nn.Sigmoid()
        else:
            # compatible with nn.CrossEntropyLoss
            self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.softmax(out)
        return out


def cnn1d_3(**kwargs):
    model = Cnn1d(**kwargs)
    return model


def cnn1d(model_name, num_classes, input_channels, pretrained=False):
    return{
        'cnn1d_3': cnn1d_3(num_classes=num_classes, input_channels=input_channels),
    }[model_name]
