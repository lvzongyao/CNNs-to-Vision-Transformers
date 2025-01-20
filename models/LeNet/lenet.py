import torch
import torch.nn as nn
import torch.nn.functional as F


# version 1, for input size(3, 3, 32, 32)
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# version 2, for input size(1, 1, 28, 28)
class LeNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.avg_pool2d(torch.sigmoid(self.conv1(x)), 2, stride=2)
        x = F.avg_pool2d(torch.sigmoid(self.conv2(x)), 2, stride=2)
        x = self.flatten(x)
        # x = x.reshape(x.size(0), -1)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


# version 3, for input (1, 1, 28, 28)
class LeNet3(nn.Module):
    def __init__(self):
        super(LeNet3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


if __name__ == '__main__':
    net = LeNet()
    net2 = LeNet2()
    net3 = LeNet3()
    net4 = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
                         nn.AvgPool2d(kernel_size=2, stride=2),
                         nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
                         nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
                         nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
                         nn.Linear(120, 84), nn.Sigmoid(), nn.Linear(84, 10))
    print(net)
    print(net2)
    print(net3)
    print(net4)
    x = torch.rand(1, 1, 28, 28)
    x2 = torch.rand(3, 3, 32, 32)
    print(net(x2))
    print(net2(x))
    print(net3(x))
    print(net4(x))
