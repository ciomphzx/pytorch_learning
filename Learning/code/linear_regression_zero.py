import torch
import random
from d2l import torch as d2l

#生成数据集
def synthetic_data(w, b, num_examples):
    #生成y = Xw + b + 噪声
    X = torch.normal(0, 1, (num_examples, len(w)))  # X.shape torch.Size([1000, 2])
    y = torch.matmul(X, w) + b  # y.shape torch.Size([1000])
    y += torch.normal(0, 0.01, y.shape)
    return X , y.reshape(-1, 1) # y.shape torch.Size([1000, 1])

w_true = torch.tensor([2, 3.4])
b_true = 4.1

# features.shape: torch.Size([1000, 2])
# labels.shape: torch.Size([1000, 1])
features, labels = synthetic_data(w_true, b_true, 1000)

# 绘制图像
# d2l.set_figsize()
# d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(),1)
# d2l.plt.show()

# 小批量读取数据集迭代生成器
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    # 生成与数据集样本数量相同的list:[0, 1, 2, ..., n]
    indices = list(range(num_examples))
    # 随机打乱顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        # 在打乱后的list中选取batch_size个样本的索引号
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)])
        # yield方法会使迭代器暂停，读取索引号对应的样本数据，下一次迭代会在暂停的位置重新开始
        yield features[batch_indices], labels[batch_indices]

#从特征和标签中小批量读取数据
# batch_size = 10
# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break


#定义模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b

class LinearRegression():
    def __init__(self, X, w, b):
        self.X = X
        self.w = w
        self.b = b
    
    def forward(self):
        return torch.matmul(self.X, self.w) + self.b
    


# 定义损失函数
def squared_loss(y_predict, y_true):
    return (y_predict - y_true.reshape(y_predict.shape)) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):
    with torch.no_grad():   # PyTorch 中用于临时关闭梯度计算的上下文管理器
        for param in params:
            # 梯度标准化，使得梯度与批次大小无关
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 初始化模型参数
w = torch.normal(0, 0.1, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 训练
lr = 1
num_epochs = 5
batch_size = 10
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        # 计算小批量样本的损失
        l = loss(net(X, w, b), y)
        # 反向传播，计算小批量样本的梯度
        l.sum().backward()
        # 更新参数
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print("epoch: ", epoch, " loss: ", float(train_l.mean()))

print('w: ', w, ' b: ', b)