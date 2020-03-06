import torch.nn as nn
import torch.nn.functional as F
import torch

def get_mask(input_batch):
    result = torch.ones(input_batch.shape)

    for i in range(input_batch.shape[0]):
        # we iterate from the end as long as we see zeros
        for j in range(input_batch.shape[2] - 1, 0, -1):
            if input_batch[i, 0, j] != 0:
                # we finally encountered real data
                break
        
            # otherwise we continue calculating mask
            result[i, 0, j] = 0

    return result


class Cnn1d(nn.Module):
    def __init__(self, num_classes=5, masking=False):
        super(Cnn1d, self).__init__()

        self.masking = masking
        if masking:
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=7, stride=1)
        else:
            self.conv1 = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=7, stride=1)
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
        masking = self.conv1.in_channels == 2
        if masking:
            mask = get_mask(x)
            x = torch.stack([x, mask], axis=1).squeeze()

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


def cnn1d(model_name, num_classes, **kwargs):
    return{
        'cnn1d_3': cnn1d_3(num_classes=num_classes, masking=kwargs.get("masking", False)),
    }[model_name]
