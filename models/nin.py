# Network in Network
import time
import torch
from torch import nn, optim
from utils import *
import sys

sys.path.append("..")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# NIN block
def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )
    return blk


# NIN Model
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Dropout(0.5),
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    # 全局平均池化层可通过将窗口形状设置成输入的高和宽实现
    # nn.AvgPool2d(kernel_size=5),
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
    # FlattenLayer()
    nn.Flatten()
)

if __name__ == '__main__':
    batch_size = 128
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size, resize=224)
    lr, num_epochs = 0.002, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
