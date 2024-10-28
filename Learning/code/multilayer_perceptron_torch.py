import torch
from torch import nn
import numpy as np
from data_loader import load_data_fashion_mnist
from soft_max_zero import train_ch3
import torch.onnx

torch.cuda.empty_cache()
## 检测是否有GPU设备 ##
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else 'cpu')

## 加载数据 ##
batch_size = 4
train_loader, test_loader = load_data_fashion_mnist(batch_size)

# 定义模型参数初始化方法
def init_weight(m):
    if m.type == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)  # 默认均值为0,方差为1
        nn.init.zeros_(m.bias)  # 偏差初始化为0

#### ---------------------------------------------------------####
## 定义模型: nn.Sequential 的实例化对象, 只适用于线性模型 ##
# net = nn.Sequential(
#     nn.Flatten(),
#     nn.Linear(28*28, 256),
#     nn.ReLU(),
#     nn.Linear(256, 10))

# # 模型初始化
# net.apply(init_weight)

# # 选择设备: GPU 或 CPU
# if device.type == 'cuda':
#     net.to(device)

# ## 定义损失函数 ##
# loss = nn.CrossEntropyLoss(reduction='none')

# ## 定义优化器 ##
# lr = 0.01
# updater = torch.optim.SGD(net.parameters(), lr)

# ## 训练 ##
# epochs =10
# train_ch3(net, train_loader, test_loader, loss, epochs, updater, device)

# ## 保存模型 ##
# torch.save(net.state_dict(), "../model/multi_layer.pth")
# print("model saved done!")

# torch.cuda.empty_cache()
# print("cuda memory release done!")

#### ---------------------------------------------------------####
## 定义模型: 用Class来定义, 灵活性更高 ##
class MultiLayer(nn.Module):
    # 初始化函数
    def __init__(self):
        super(MultiLayer, self).__init__()  # 为了兼容python2的写法
        # super().__init__()    # python3的简洁写法
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28*28, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 10)
    #　前向传播函数
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x    # 返回模型的输出

# 实例化类
model = MultiLayer()
# 参数初始化
model.apply(init_weight)
#　模型转移到GPU
if device.type == 'cuda':
    model.to(device)

# 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 定义优化器
lr = 0.01   # 学习率
updater = torch.optim.SGD(model.parameters(), lr)

# 训练
epoch = 10
# 传入参数为 model , 前向传播会隐式地调用 forward() 方法
train_ch3(model, train_loader, test_loader, loss, epoch, updater, device)

## 保存模型 ##
torch.save(model.state_dict(), "../model/pth/model.pth")
print("model saved done!")

torch.cuda.empty_cache()
print("cuda memory release done!")