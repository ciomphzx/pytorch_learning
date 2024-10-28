import torch
from torch import nn
import numpy as np
from torch.utils import data
from d2l import torch as d2l
import torch.onnx

# 构建模拟数据集
w_true = torch.tensor([2.3, 3.4])
b_true = 4.5
features, labels = d2l.synthetic_data(w_true, b_true, 1000)

# 构建数据加载器
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

# 定义模型
net = nn.Sequential(nn.Linear(2, 1))

# 初始化模型参数
net[0].weight.data.normal_(0, 0.02)    #权重
net[0].bias.data.fill_(0)   #偏置

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法：实例化一个SGD对象，SGD类是optim类的子类
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练过程
num_epochs = 5
batch_size = 10
data_iter = load_array((features, labels), batch_size=batch_size)
for epoch in range (num_epochs):
    for X, y in data_iter:
        # 计算损失
        l = loss(net(X), y)
        # 梯度清零，可以调用基类的方法
        trainer.zero_grad()
        # 反向传播
        l.backward()
        # 更新参数
        trainer.step()
    l = loss(net(features), labels)
    print('epoch: ', epoch + 1, 'loss: ', l)


# 创建一个虚拟的输入张量
dummy_input = torch.randn(1, 2)  # 批量大小为1，特征维度为2

# 指定ONNX文件的保存路径
onnx_file_path = "../model/linear_regression.onnx"

# 导出模型为ONNX格式
torch.onnx.export(
    net,  # 要导出的模型
    dummy_input,  # 模型的输入张量
    onnx_file_path,  # 输出的ONNX文件路径
    input_names=['input'],  # 输入张量的名字
    output_names=['output'],  # 输出张量的名字
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # 支持动态批量大小
)
