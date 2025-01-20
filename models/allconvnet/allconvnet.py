# https://arxiv.org/abs/1412.6806 "Striving for Simplicity: The All Convolutional Net"
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')
import utils_for_google_drive


class AllConvNet(nn.Module):
    def __init__(self, in_channels, num_classes=10, dropout=True):
        super().__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(in_channels, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, stride=2, padding=1)
        # self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv7 = nn.Conv2d(192, 192, 3)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.conv_class = nn.Conv2d(192, num_classes, 1)

    def forward(self, x):
        if self.dropout:
            x = F.dropout(x, .2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        if self.dropout:
            x = F.dropout(x, .5)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        if self.dropout:
            x = F.dropout(x, .5)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv_class(x))

        x = x.reshape(x.size()[0], x.size()[1], -1).mean(-1)
        # x = F.adaptive_avg_pool2d(x, 1)
        # x.squeeze_(-1)
        # x.squeeze_(-1)
        return x
