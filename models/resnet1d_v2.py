import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=5, stride=stride, padding=2)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet1d_v2(nn.Module):
    def __init__(self, block, layers, num_classes=5, input_channels=1):
        self.inplanes = 32
        super(ResNet1d_v2, self).__init__()

        self.conv1 = nn.Conv1d(
            input_channels, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 254, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=(1))
        self.fc = nn.Linear(254 * block.expansion, num_classes)

        if 1 == num_classes:
            # compatible with nn.BCELoss
            self.softmax = nn.Sigmoid()
        else:
            # compatible with nn.CrossEntropyLoss
            self.softmax = nn.LogSoftmax(dim=1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if not (stride == 1 and self.inplanes == planes * block.expansion):
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = self.softmax(x)
        return x


def resnet1d_v2_10(pretrained=False, **kwargs):
    model = ResNet1d_v2(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet1d_v2_18(pretrained=False, **kwargs):
    model = ResNet1d_v2(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet1d_v2(model_name, num_classes, **kwargs):
    return{
        'resnet1d_v2_18': resnet1d_v2_18(num_classes=num_classes, input_channels=1),
        'resnet1d_v2_10': resnet1d_v2_10(num_classes=num_classes, input_channels=1),
    }[model_name]
