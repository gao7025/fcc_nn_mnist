# -*- coding: utf-8 -*-
import torch
import torchvision
from torchvision import datasets, transforms

"""
method 1: load existing dataset
"""
# 定义转换操作
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化
])

# 加载训练数据集
train_dataset = datasets.MNIST(root='mnist_data', train=True, download=False, transform=transform)
# 加载测试数据集
test_dataset = datasets.MNIST(root='mnist_data', train=False, download=False, transform=transform)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

print(train_dataset, train_loader)


"""
method 2: load online dataset
定义Dataset与DateLoader
2.1dataset类对数据进行统一封装，本质是一个表示数据集的抽象类，里面有三个函数:init,getitem,len三个函数。
2.2dataloader则是加载dataset,并设置其batch_size（单次训练时送入的样本数目），以及shuffle(是否打乱样本顺序，训练集一定要设置shuffle为True，测试集则无强制性规定)
2.3Compose把多种数据处理的方法集合在一起。将需要被处理的数据转为Tensor类型，数据被转化为Tensor类型后，对其进行减均值（0.1307）和除标准差（0.3081）,以实现数据的正则化。
"""
batch_size = 512
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)

print(train_dataset, train_loader)

