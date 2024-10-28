#### 该代码从零实现了softmax,并对模型进行了训练 ####
import torch
from IPython import display
from d2l import torch as d2l
from data_loader import load_data_fashion_mnist   # 从Data_Loader.py中加载所有函数

# 在n个变量上累加
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):   # 将self.data和args两个list对应位置的元素相加, 值赋给self.data
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
# 加载数据 #
batch_size = 256
trainer_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

# 初始化模型参数 #
# 输入Tensor的维度:图像宽*高*通道数
input_num = 28 * 28 * 1
# 输出Tensor的维度:类别
output_num = 10
# 初始化权重参数
W = torch.normal(0, 0.01, size=[input_num, output_num], requires_grad=True)
# 初始化偏置参数
b = torch.zeros(output_num, requires_grad=True)

# 定义softmax操作, 对前层的输出进行softmax操作
def softmax(X):
    X_exp = torch.exp(X)    # X.shape & X_exp.shape: torch.Size([batch_size, output_szie])
    # 对每行元素求和，保持行数不变
    partition = X_exp.sum(1, keepdim=True)  # partition.shape: torch.size([batch_size, 1])
    return X_exp / partition    # (return value).shape: torch.size([batch_size, output_szie])

# 定义网络模型
def net(X):
    # X.shape:torch.Size([batch_size, C, W, H])
    # W.shape:torch.Size([C*W*H, output_size]) W.shape[0]:C*W*H
    # X.reshape(-1, W.shape[0]): torch.size([batch_size, C*W*H])
    # [batch_size, C*W*H] * [C*W*H, output_size] = [batch_size, output_size]
    # (return value).shape: [batch_size, output_size], 模型输出预测值的维度
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)

# 定义损失函数:交叉熵损失
def loss(y_hat, y):
    # y_hat.shape: torch.Size([batch_size, output_size])
    # y.shape: torch.Size([batch_size])
    # (y_hat[range(len(y_hat)), y]).shape: torch.Size([batch_size])
    # 从模型输出的one-hot向量中获得标签对应类别模型的预测值, 即y_hat[i][y[i]], 然后计算交叉熵损失
    return - torch.log(y_hat[range(len(y_hat)), y])

lr = 0.01
def updater(batch_size):
    """优化器
    Args:
        batch_size : 每批次数据量
    """
    return d2l.sgd([W, b], lr, batch_size)

# 定义计算正确预测目标数函数
def accuracy(y_hat, y):
    if(len(y_hat.shape) > 1 and y_hat.shape[1] > 1):
        # 求y_hat每一行最大值元素所在索引位置, y_hat.shape: torch.Size([batch_size, output_size])
        y_hat = y_hat.argmax(axis=1)    # y_hat.shape: torch.Size([batch_size])
        cmp = y_hat.type(y.dtype) == y  # 索引位置与标签类别进行匹配, 正确为True, 错误为False
    return float(cmp.type(y.dtype).sum())   # 计算cmp中True元素的总数, 即为正确预测的目标数量

def evaluate_accuracy(net, data_iter, device):
    """计算模型在指定数据集的精度
    Args:
        net: 模型
        data_iter: 数据加载器

    Returns:
        float: 模型精度
    """
    if isinstance(net, torch.nn.Module):
        net.eval() # 将模型设置为评估模式
    metric = Accumulator(2) # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            if device.type == 'cuda':
                X, y = X.to(device), y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 训练模型一个迭代周期epoch, 返回模型在训练集上的损失和精度
def train_epoch_ch3(net, train_iter, loss, updater, device):
    """在数据集迭代训练模型一个周期
    Args:
        net : 网络模型
        train_iter : 训练集数据加载器
        loss : 损失函数
        updater : 优化器
    """
    # 将模型设置为训练模式  
    if isinstance(net, torch.nn.Module):    # 判断net是不是Module的实例对象
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    # 根据一个批次的数据计算梯度并更新参数; 
    # X.shape:torch.Size([batch_size, C, W, H]) y.shape: torch.Size([batch_size])
    for X, y in train_iter:
        if device.type == 'cuda':
            X, y = X.to(device), y.to(device)
        y_hat = net(X)  # y_hat.shape: torch.Size([batch_size, output_size])
        l = loss(y_hat, y) # l.shape: torch.Size([batch_size])
        # isinstance判断对象是否是Class的实例化对象
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad() #梯度清零
            l.mean().backward()
            updater.step()
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()  # 方向传播计算梯度, 梯度存储在.grad属性
            updater(X.shape[0]) # 更新参数
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度, 损失值已经用样本总数进行了归一化，平均在每个样本上的损失         
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater, device='cpu'): 
    """在数据集上训练模型多个epoch
    Args:
        net: 网络模型
        train_iter: 训练集数据加载器
        test_iter: 测试集数据加载器
        loss: 损失函数
        num_epochs: 训练周期数
        updater: 优化器
    """
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],  legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater, device)
        test_acc = evaluate_accuracy(net, test_iter, device)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        train_loss, train_acc = train_metrics
        print(f'Epoch {epoch + 1}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# 训练
# num_epochs = 10
# train_ch3(net, trainer_iter, test_iter, loss, num_epochs, updater)

