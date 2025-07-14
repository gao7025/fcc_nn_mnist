# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from utils import plot_image, plot_curve, one_hot
from torchvision import datasets, transforms

# 定义转换操作
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

# 加载训练和测试数据集
train_dataset = datasets.MNIST(root='mnist_data', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root='mnist_data', train=False, download=False, transform=transform)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


batch_size = 512


x, y = next(iter(train_loader))
print(x.shape, y.shape, x.min(), x.max())
plot_image(x, y, 'image sample')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # xw+b
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    # 定义数据如何通过网络流动
    def forward(self, x):
        # x: [b, 1, 28, 28]
        # h1 = relu(xw1+b1)
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w2+b2)
        x = F.relu(self.fc2(x))
        # h3 = h2w3+b3
        x = self.fc3(x)

        return x


# 创建这个网络架构的实例
net = Net()

# [w1, b1, w2, b2, w3, b3]

# 训练网络
# 接下来我们需要设置一个优化器（optimizer）和一个损失准则（loss criterion）
# 创建一个随机梯度下降（stochastic gradient descent）优化器，
# 第一个参数包括权重w，和偏置b等是神经网络中的参数，
# 第二个参数lr是学习率。sgd中的学习率lr的作用可以理解为p′ = p − lr ∗ dp
# 第三个参数momentum是冲量,在普通的梯度下降法x+=v中，每次x的更新量v为v=−dx∗lr，其中dx为目标函数func(x)对x的一阶导数。
# 当使用冲量时，则把每次x的更新量v考虑为本次的梯度下降量−dx∗lr与上次x的更新量v乘上一个介于[0,1][0,1]的因子momentum的和，
# 即v′ = − dx ∗ lr + v ∗ momemtum
# 当本次梯度下降- dx * lr的方向与上次更新量v的方向相同时，上次的更新量能够对本次的搜索起到一个正向加速的作用。
# 当本次梯度下降- dx * lr的方向与上次更新量v的方向相反时，上次的更新量能够对本次的搜索起到一个减速的作用。
# 第四个参数weight_decay是权重衰减,即L2正则化前面的那个λ参数， 权重衰减的使用既不是为了提高你所说的收敛精确度也不是为了提高收敛速度，
# 其最终目的是防止过拟合。在损失函数中，weight decay是放在正则项（regularization）前面的一个系数，正则项一般指示模型的复杂度，
# 所以weight decay的作用是调节模型复杂度对损失函数的影响，若weight decay很大，则复杂的模型损失函数的值也就大。

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# # 创建一个损失函数
# criterion = nn.NLLLoss()
train_loss = []

for epoch in range(3):

    for batch_idx, (x, y) in enumerate(train_loader):

        # x: [b, 1, 28, 28], y: [512]
        # [b, 1, 28, 28] => [b, 784]
        x = x.view(x.size(0), 28*28)
        # => [b, 10]
        # 调用Net类中的forward()方法
        out = net(x)
        # [b, 10]
        y_onehot = one_hot(y)
        # loss = mse(out, y_onehot)
        loss = F.mse_loss(out, y_onehot)
        # 梯度归零/重置
        optimizer.zero_grad()
        loss.backward()
        # w' = w - lr*grad
        optimizer.step()

        train_loss.append(loss.item())

        # if batch_idx % 10 == 0:
        #     print(epoch, batch_idx, loss.item())
        log_interval = 100
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

plot_curve(train_loss)
# we get optimal [w1, b1, w2, b2, w3, b3]


total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 28*28)
    out = net(x)
    # out: [b, 10] => pred: [b]
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc:', acc)

x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28*28))
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')

