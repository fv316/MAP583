import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, num_classes=5, input_channels=1):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=8, kernel_size=7, stride=1)
        self.conv2 = nn.Conv1d(8, 32, kernel_size=7, stride=1)
        self.fc1   = nn.Linear(in_features=576, out_features=num_classes)

        if 1 == num_classes: 
            # compatible with nn.BCELoss
            self.softmax = nn.Sigmoid()
        else:    
            # compatible with nn.CrossEntropyLoss
            self.softmax = nn.LogSoftmax(dim=1)        


    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool1d(out, kernel_size=3, stride=3)
        out = F.relu(self.conv2(out))
        out = F.max_pool1d(out, kernel_size=3, stride=3)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.softmax(out)
        return out


def cnn1d_3( **kwargs):
    model = CNN1D(**kwargs)
    return model


def cnn1d(model_name, num_classes, input_channels, pretrained=False):
    return{
        'cnn1d_3': cnn1d_3(num_classes=num_classes, input_channels=input_channels),
    }[model_name]