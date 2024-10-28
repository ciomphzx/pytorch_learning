#### 此文件是如何初始化深度神经网络中线性层的权重 ####
import torch
from torch import nn

## 定义神经网络模型 ##
class Linear_Module(nn.Module): # 继承自nn.Module类
    ## 初始化 ##
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 8)
        self.linear2 = nn.Linear(8, 4)
        self.relu = nn.ReLU()
    ## 前向传播 ##
    def forward(self, x):   # 参数中要有self, 指代模型本身
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# 实例化模型
model = Linear_Module()

## 定义模型参数初始化方法:随机初始化 ##
def init_weight_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)   #默认均值为0,方法为1
        nn.init.zeros_(m.bias)

# 使用随机初始化的方式来初始化模型权重参数
model.apply(init_weight_normal)
print(model.linear1.weight.shape)   # 模型权重的size
print(model.linear1.weight.data)    # 模型是自定义的类, 要通过属性名来访问其参数
print(model.linear1.bias.shape) # 模型偏置的size
print(model.linear1.bias.data)  # 模型偏置

## 将模型权重初始化为常数 ##
def init_weight_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)    # 将权重初始化为1
        nn.init.zeros_(m.bias)

# 将模型权重初始化为1
model.apply(init_weight_constant)
# 模型是自定义的类, 要通过属性名来访问其参数
print(model.linear1.weight.shape)   # 模型第一个线性层权重的size
print(model.linear1.weight.data)    # 模型第一个线性层权重
print(model.linear1.bias.shape) # 模型第一个线性层偏置的size
print(model.linear1.bias.data)  # 模型第一个线性层偏置

## Xavier 初始化模型权重参数 ##
def init_weight_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)    # Xavier初始化
        nn.init.zeros_(m.bias)

# 将模型权重参数按 Xavier 方法进行初始化
model.apply(init_weight_xavier)
# 模型是自定义的类, 要通过属性名来访问其参数
print(model.linear1.weight.shape)   # 模型第一个线性层权重的size
print(model.linear1.weight.data)    # 模型第一个线性层权重
print(model.linear1.bias.shape) # 模型第一个线性层偏置的size
print(model.linear1.bias.data)  # 模型第一个线性层偏置

## He 初始化模型权重参数 ##
def init_weight_kaiming(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

# 将权重参数按照 He 方法初始化
model.apply(init_weight_kaiming)
# 模型是自定义的类, 要通过属性名来访问其参数
print(model.linear1.weight.shape)   # 模型第一个线性层权重的size
print(model.linear1.weight.data)    # 模型第一个线性层权重
print(model.linear1.bias.shape) # 模型第一个线性层偏置的size
print(model.linear1.bias.data)  # 模型第一个线性层偏置

## 保存模型 ##
torch.save(model.state_dict(), "../model/pth/linear_model.pth")