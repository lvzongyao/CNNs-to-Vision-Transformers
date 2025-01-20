# Another version of VGGNet implementation

from utils import *
import time
import torch
from torch import nn, optim
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# vgg block
def vgg_block(num_convs, in_channels, out_channels):
    block = []
    for i in range(num_convs):
        if i == 0:
            block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            block.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        block.append(nn.ReLU())
    block.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*block)


# vgg network version 1
def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        # 每经过一个vgg_block都会使宽高减半
        net.add_module('vgg_block_' + str(i + 1), vgg_block(num_convs, in_channels, out_channels))
    # 全连接层部分
    net.add_module('fc', nn.Sequential(# FlattenLayer(),
                                       nn.Flatten(),
                                       nn.Linear(fc_features, fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, 10)
                                       ))
    return net

# # 构造一个高和宽均为224的单通道数据样本来观察每一层的输出形状
# net = vgg(conv_arch, fc_features, fc_hidden_units)
# x = torch.rand(1, 1, 224, 224)
#
# # named_children获取一级子模块及其名字(named_modules会返回所有子模块,包括子模块的子模块)
# for name, blk in net.named_children():
#     x = blk(x)
#     print(name, 'output shape: ', x.shape)


# vgg network version 2
def vgg2(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    net = nn.Sequential(*conv_blks, nn.Flatten(),
                        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
                        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                        nn.Linear(4096, 10))
    return net


# vgg network version 3
vgg_net = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),

                        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),

                        nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),

                        nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),

                        nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),

                        nn.Flatten(),
                        nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
                        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                        nn.Linear(4096, 10))


if __name__ == '__main__':
    # training
    # vgg net structure
    # 设定卷积块参数
    conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
    # 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7
    fc_features = 512 * 7 * 7  # c * w * h
    fc_hidden_units = 4096  # 任意

    ratio = 8
    small_conv_arch = ((1, 1, 64 // ratio), (1, 64 // ratio, 128 // ratio),
                       (2, 128 // ratio, 256 // ratio), (2, 256 // ratio, 512 // ratio),
                       (2, 512 // ratio, 512 // ratio))
    net = vgg(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)
    # print(net)

    conv_arch2 = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    net2 = vgg2(conv_arch2)
    # print(net2[0])
    # print('\n')
    # print(net2)
    small_conv_arch2 = [(pair[0], pair[1] // ratio) for pair in conv_arch2]
    # print(small_conv_arch2)
    net2_small = vgg2(small_conv_arch2)
    # print(net2_small)

    batch_size = 64
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
