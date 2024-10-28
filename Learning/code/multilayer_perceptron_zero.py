#### 多层感知机从零实现 ####
import torch
from torch import nn
from data_loader import load_data_fashion_mnist
from soft_max_zero import train_ch3

torch.cuda.empty_cache()
## 检测是否有GPU ##
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else 'cpu')

## 加载数据 ##
batch_size = 64
train_loader, test_loader = load_data_fashion_mnist(batch_size)

## 初始化模型参数 ##
num_inputs = 28 * 28
num_hiddens = 256
num_outs = 10
# 隐藏层参数
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
# 输出层参数
W2 = nn.Parameter(torch.randn(num_hiddens, num_outs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outs, requires_grad=True))
Params = [W1, b1, W2, b2]

## 激活函数 ##
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

## 定义模型 ##
def net(X):
    if device:
        X = X.reshape(-1, num_inputs)
        H = relu(torch.matmul(X, W1.to(device)) + b1.to(device))
    return torch.matmul(H, W2.to(device)) + b2.to(device)


## 损失函数 ##
loss = nn.CrossEntropyLoss(reduction='none')

## 优化器 ##
lr = 0.01
updater = torch.optim.SGD(Params, lr)

## 训练 ##
epochs = 10
train_ch3(net, train_loader, test_loader, loss, epochs, updater, device)