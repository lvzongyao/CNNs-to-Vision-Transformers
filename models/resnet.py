import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# residual block
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use1_1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use1_1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        return F.relu(y + x)


# resnet model
net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    # 对第一个模块做了特别处理
    if first_block:
        assert in_channels == out_channels  # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use1_1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


# 加入全部4个模块,每个模块使用2个残差块（简单起见）
net.add_module('resnet_block1', resnet_block(64, 64, 2, first_block=True))
net.add_module('resnet_block2', resnet_block(64, 128, 2))
net.add_module('resnet_block3', resnet_block(128, 256, 2))
net.add_module('resnet_block4', resnet_block(256, 512, 2))

# 与GoogLeNet一样，加入全局平均池化层后接上全连接层输出
net.add_module('global_avg_pool', nn.AdaptiveAvgPool2d((1, 1)))  # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
net.add_module('fc', nn.Sequential(nn.Flatten(), nn.Linear(512, 10)))

# # 查看模块形状
# x = torch.rand((1, 1, 224, 224))
# for name, layer in net.named_children():
#     x = layer(x)
#     print(name, 'output shape:\t', x.shape)


if __name__ == '__main__':
    # 训练
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96, root='E:\Datasets\FashionMNIST')

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
