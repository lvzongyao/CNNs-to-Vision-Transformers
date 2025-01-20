## Purpose

This repository contains implementations of popular Convolutional Neural Network (CNN) backbone networks for image classification, object detection, and segmentation tasks. The goal is to provide easy-to-use and well-documented implementations of these models for researchers and practitioners.

## Usage

To use the code in this repository, follow the instructions below to install dependencies, run the models, and explore the different implementations.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lvzongyao/cnn-classification-networks.git
   cd cnn-classification-networks
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

<!--
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
-->

### Running the Models

To run the models, you can use the provided `train.py` script. Here are some examples:

1. Train a ResNet-34 model on the CIFAR-10 dataset:
   ```bash
   python train.py --dataset cifar10 --data_path /path/to/cifar10 --num_classes 10 --batch_size 128 --model resnet34 --num_epochs 30
   ```

2. Train a ResNet-34 model on the CIFAR-100 dataset:
   ```bash
   python train.py --dataset cifar100 --data_path /path/to/cifar100 --num_classes 100 --batch_size 128 --model resnet34 --num_epochs 30
   ```

3. Train ViT on MNIST:
   ```bash
   python vit_train.py --dataset mnist --epochs 10 --batch_size 128 --lr 0.001 --device cuda --num_classes 10
   ```

4. Train ViT on CIFAR-10:
   ```bash
   python vit_train.py --dataset cifar10 --epochs 20 --batch_size 64 --lr 0.0001 --device cuda --num_classes 10
   ```

3. Train ViT on CIFAR-100:
   ```bash
   python vit_train.py --dataset cifar100 --epochs 30 --batch_size 32 --lr 0.0005 --device cuda --num_classes 100
   ```

## Models

This repository includes implementations of the following models:

1. **LeNet**: A classic convolutional neural network for image classification. 
   - Implementation: `models/LeNet/lenet.py`
   - Training script: `models/LeNet/train.py`

2. **AlexNet**: A deep convolutional neural network for image classification. 
   - Implementation: `models/alexnet.py`

3. **Network in Network (NIN)**: A network that replaces traditional convolutional layers with micro neural networks. 
   - Implementation: `models/nin.py`
   - 
4. **VGG**: A deep convolutional neural network with very small convolutional filters. 
   - Implementation: `models/vgg.py`
   - Alternative implementation: `models/vgg_2.py`
     
5. **AllConvNet**: An all convolutional network for image classification. 
   - Implementation: `models/allconvnet/allconvnet.ipynb`
   - Python file: `models/allconvnet/allconvnet.py`

6. **ResNet**: A deep residual network for image classification. 
   - Implementation: `models/resnet.py`

7. **Vision Transformer (ViT)**: A transformer-based model for image classification.
   - Implementation: `models/vit.py`

## Directory Structure

The repository is organized as follows:

```
CNNs-to-Vision-Transformers/
├── models/
│   ├── LeNet/
│   │   ├── lenet.py
│   │   ├── train.py
│   ├── allconvnet/
│   │   ├── allconvnet.ipynb
│   │   ├── allconvnet.py
│   │   ├── best.pt
│   │   ├── checkpoint/
│   │   │   ├── allconvnet_cifar10_epoch_1.ckpt
│   │   │   ├── allconvnet_cifar10_epoch_2.ckpt
│   ├── alexnet.py
│   ├── nin.py
│   ├── resnet.py
│   ├── vgg.py
│   ├── vgg_2.py
│   ├── vit.py
├── train.py
├── utils.py
├── utils_for_google_drive.py
├── commands.txt
├── .gitignore
└── README.md
```

## Examples and Explanations

### LeNet \\[paper\\](https://ieeexplore.ieee.org/document/726791) (1998)
### LeNet [paper](https://ieeexplore.ieee.org/document/726791) (1998)

The LeNet model is a classic convolutional neural network for image classification. It was one of the first successful applications of CNNs and is widely used for educational purposes.

### AlexNet (2012)

The AlexNet model is a deep convolutional neural network for image classification. It was one of the first models to achieve significant performance improvements on the ImageNet dataset.

### Network in Network (NIN) (2013)

The Network in Network (NIN) model replaces traditional convolutional layers with micro neural networks, which helps to improve the model's performance and efficiency.

### VGG (2014)

The VGG model is a deep convolutional neural network with very small convolutional filters. It is known for its simplicity and effectiveness in image classification tasks.

### AllConvNet (2015)

The AllConvNet model is an all convolutional network for image classification. It replaces traditional fully connected layers with convolutional layers, making the network more efficient and easier to train.

### ResNet (2015)

The ResNet model is a deep residual network for image classification. It introduces residual connections, which help to mitigate the vanishing gradient problem and enable the training of very deep networks.

### Vision Transformer (ViT)

The Vision Transformer (ViT) is a deep learning architecture that applies the transformer model, originally designed for natural language processing, to image recognition tasks by dividing images into patches and processing them as sequences, achieving state-of-the-art performance on various vision benchmarks.
