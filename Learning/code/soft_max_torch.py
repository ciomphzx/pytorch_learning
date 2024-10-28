### 该代码是利用pytorch的API来实现softmax ###
import torch
from torch import nn
from Data_Loader import *
from soft_max_zero import train_ch3

# 检查是否有GPU
flag_GPU = torch.cuda.is_available()
if flag_GPU:
    print(f'GPU count: {torch.cuda.device_count()}')    # GPU数量
# 选择GPU设备, 默认cuda 0(第一个GPU设备)
device = torch.device('cuda' if flag_GPU else 'cpu')

# 加载数据集
batch_size = 512
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

# 定义模型
image_width = 28
image_height = 28
image_channel = 1
num_class = 10

num_input = image_channel * image_width * image_height
num_output = num_class
net = nn.Sequential(nn.Flatten(), nn.Linear(num_input, num_output))

# 模型参数初始化函数
def init_weight(m):
    # 线性层使用随机初始化
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01) # 默认均值为0, 方差为1
        

# 初始化模型参数
net.apply(init_weight)
if device.type == 'cuda':
    net.to(device)
# 将softMax与损失函数在一起实现
loss = nn.CrossEntropyLoss(reduction='none')

# 定义优化器
updater = torch.optim.SGD(net.parameters(), lr=0.01)

# 训练
num_epoch = 10
train_ch3(net, train_iter, test_iter, loss, num_epoch, updater, device)
    