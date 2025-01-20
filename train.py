import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models, datasets, transforms

import argparse
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpu-id', type=int, default=0,
                            help='id of gpu to use ' '(only applicable to non-distributed training)')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Name of the dataset used')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to where the data is')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes of the dataset')
    parser.add_argument('--num_channels', type=int, default=3, help='number of channels of the dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size used for training and testing')
    parser.add_argument('--model', type=str, default='resnet18', help='Name of the model used')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--latent_dim', type=int, default=32, help='The dimensionality of the VAE latent dimension')
    parser.add_argument('--beta', type=float, default=1, help='Hyperparameter for training. The parameter for VAE')
    parser.add_argument('--num_adv_steps', type=int, default=1,
                        help='Number of adversary steps taken for every task model step')
    parser.add_argument('--num_vae_steps', type=int, default=2,
                        help='Number of VAE steps taken for every task model step')
    parser.add_argument('--adversary_param', type=float, default=1,
                        help='Hyperparameter for training. lambda2 in the paper')
    parser.add_argument('--out_path', type=str, default='./results', help='Path to where the output log will be')
    parser.add_argument('--log_name', type=str, default='accuracies.log',
                        help='Final performance of the models will be saved with this name')
    args = parser.parse_args()

    return args


def imagenet_train_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def imagenet_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def mnist_train_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def mnist_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def cifar10_train_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                      std=[0.5, 0.5, 0.5])
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def cifar10_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def cifar100_train_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])


def cifar100_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])


def get_dataset(args):
    if args.dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=False,
                                                 transform=cifar10_train_transform())
        test_set = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=False,
                                                transform=cifar10_test_transform())
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    if args.dataset == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=False,
                                                  transform=cifar100_train_transform())
        test_set = torchvision.datasets.CIFAR100(root=args.data_path, train=False, download=False,
                                                 transform=cifar100_test_transform())
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    if args.dataset == 'tinyimagenet':
        train_set = datasets.ImageFolder(root=args.data_path, transform=imagenet_train_transform())
        test_set = datasets.ImageFolder(root=args.data_path, transform=imagenet_test_transform())

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    if args.dataset == 'imagenet':
        train_set = datasets.ImageFolder(root=args.data_path, transform=imagenet_train_transform())
        test_set = datasets.ImageFolder(root=args.data_path, transform=imagenet_test_transform())

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    return train_set, test_set, train_loader, test_loader


def get_model(args):
    """ return model architecture for image recognition
    Args:
        name: which model architecture to use
        num_classes: number of classes for classification
        pretrained: whether or not to get imagenet pretrained model
    Return:
        model: neural network object to be used
    """
    if args.model == 'resnet18':
        model = models.resnet18
        # model = resnet18
    elif args.model == 'resnet34':
        model = models.resnet34
    elif args.model == 'resnet50':
        model = models.resnet50
        # model = resnet50
    elif args.model == 'resnet101':
        model = models.resnet101
        # model = resnet101
    # elif name == 'resnet110':
    #     model = resnet110
    elif args.model == 'resnet152':
        model = models.resnet152
        # model = resnet152

    return model(num_classes=args.num_classes)


def evaluate_accuracy(model, data_loader, device=None):
    model.eval()  # evaluation mode, turn off dropout
    if device is None and isinstance(model, torch.nn.Module):
        # use net.device if no designated device
        device = list(model.parameters())[0].device

    correct, total, loss = 0.0, 0, 0.0
    with torch.no_grad():
        for imgs, labels in data_loader:
            if isinstance(model, torch.nn.Module):
                # acc_sum += (model(imgs.to(device)).argmax(dim=1) == labels.to(device)).float().sum().cpu().item()
                output = model(imgs.to(device))
                preds = output.argmax(dim=1)
                correct += (preds == labels.to(device)).cpu().sum().item()
            total += labels.shape[0]
    model.train()  # back to training mode
    return correct / total, loss / len(data_loader)


def evaluate_accuracy_with_test_loss(model, data_loader, criterion=None, device=None):
    model.eval()  # evaluation mode, turn off dropout
    if device is None and isinstance(model, torch.nn.Module):
        # use net.device if no designated device
        device = list(model.parameters())[0].device

    correct, total, loss = 0.0, 0, 0.0
    with torch.no_grad():
        for imgs, labels in data_loader:
            if isinstance(model, torch.nn.Module):
                # acc_sum += (model(imgs.to(device)).argmax(dim=1) == labels.to(device)).float().sum().cpu().item()
                output = model(imgs.to(device))
                preds = output.argmax(dim=1)
                correct += (preds == labels.to(device)).cpu().sum().item()
                # loss += F.nll_loss(output, labels.to(device)).cpu().item()
                loss += criterion(output, labels.to(device)).cpu().item()
            total += labels.shape[0]
    model.train()  # back to training mode
    return correct / total, loss / len(data_loader)


def train(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, total, start = 0.0, 0.0, 0, time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(imgs)

            loss = criterion(output, labels)
            loss.backward()

            optimizer.step()

            train_loss_sum += loss.cpu().item()
            train_acc_sum += (output.argmax(dim=1) == labels).sum().cpu().item()
            total += labels.shape[0]

        torch.save(model.state_dict(),
                   './checkpoints/' + args.model + '_' + args.dataset + '_' + time.strftime('%m%d_%H_%M_%S') + '.pt')
        # model.load_state_dict(torch.load('./checkpoints/resnet18_cifar10.pt'))

        test_acc, test_loss = evaluate_accuracy(model, test_loader, device)
        # test_acc, test_loss = evaluate_accuracy_with_test_loss(model, test_loader, criterion, device)

        print('Epoch %d: training loss %.4f, train acc %.3f, test acc %.3f, test loss %.4f, time %.1f sec'
              % (epoch + 1, train_loss_sum / len(train_loader), train_acc_sum / total,
                 test_acc, test_loss, time.time() - start))


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set, test_set, train_loader, test_loader = get_dataset(args)
    model = get_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    print("Training on ", torch.cuda.get_device_name(0))
    train(model, train_loader, test_loader, criterion, optimizer, args.num_epochs, device)
