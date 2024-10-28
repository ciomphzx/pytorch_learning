#### 此文件中是定义数据加载器，小批量读取数据 ####
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

# print(torch.cuda.is_available())

#### 数据集并读取到内存中; 保存到文件夹下 ####
# ToTensor类：将Image转换为tensor, 其形状为[C, W, H], 并且将像素值归一化为0到1
# trans是ToTensor类实例化的一个对象，trans可以像函数一样被调用，实际上调用的是类定义好的__call__函数
# trans = transforms.ToTensor()
# # # transform参数是要对加载图像数据执行的转换操作
# mnist_train = torchvision.datasets.FashionMNIST(
#     root='../dataset/FashionMNIST/train', train=True, 
#     transform=trans, download=True
# )
# mnist_test = torchvision.datasets.FashionMNIST(
#     root='../dataset/FashionMNIST/test', train=False,
#     transform=trans, download=True
# )

# print('train set num: ', len(mnist_train))
# print('test set num: ', len(mnist_test))
# print('image size is ', mnist_train[0][0].shape)

# # # 读取图像标签
# def get_labels(labels):
#     text_labels = ['t-shirt', 'trouser', 'pullover', 'dress',
#                    'coat',  'sandal', 'shirt', 'sneaker',
#                    'bag', 'ankle boot']
#     return [text_labels[int(i)] for i in labels]

# # # 可视化样本
# def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
#     figsize = (num_cols * scale, num_cols * scale)
#     _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
#     axes = axes.flatten()
#     for i, (ax, img) in enumerate(zip(axes, imgs)):
#         if torch.is_tensor(img):
#             ax.imshow(img.numpy())
#         else:
#             ax.imshow(img)
#         ax.axes.get_xaxis().set_visible(False)
#         ax.axes.get_yaxis().set_visible(False)
#         if titles:
#             ax.set_title(titles[i])
#     d2l.plt.show()
#     return axes

# X, y = next(iter(data.DataLoader(mnist_train, batch_size=20)))
# show_images(X.reshape(20, 28, 28), 2, 10, titles=get_labels(y))

# # # 加载数据:随机小批量
# trainer_iter = data.DataLoader(mnist_train, batch_size=256, shuffle=True, num_workers=0)

# # # 测试数据加载的时间
# timer = d2l.Timer()
# for X, y in trainer_iter:   #实际调用的是__iter__方法
#     continue
# print("Time: ", timer.stop(), " s")

# 数据加载器函数：加载并预处理 Fashion-MNIST 数据集，将其转化为 PyTorch 支持的批次数据加载器#
def load_data_fashion_mnist(batch_size, resize=None):
    """Fashion-MNIST 数据集的数据加载器函数
    Args:
        batch_size (_type_): 批量大小
        resize (_type_, optional): 是否resize, 默认没有.

    Returns:
        DataLoader实例对象: Fashion-MNIST 训练集、测试集的数据加载器
    """
    # 定义了一个转换操作的列表，并且列表中有一个操作(Image转换为Tensor,并且像素值归一化)
    trans = [transforms.ToTensor()]
    if resize:  
        # 将图像调整到指定大小，然后将其插入到转换列表的第一位, 这就保证了先resize后转换为Tensor
        trans.insert(0, transforms.Resize(resize))
    # 将对图像进行的操作进行整合
    trans = transforms.Compose(trans)
    # transform=trans 执行trans列表中定义图像的所有转换操作(resize和ToTensor)
    # 返回值的类型是tuple: {image_data, label}
    mnist_train = torchvision.datasets.FashionMNIST(root="../dataset/FashionMNIST/train", train=True,
                                                    transform=trans, download=True)
    # print(mnist_train[0][0], mnist_train[0][1])
    mnist_test  = torchvision.datasets.FashionMNIST(root="../dataset/FashionMNIST/test", train=False,
                                                    transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, 
                            num_workers=0),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=0))

# 从数据加载器中读取图像数据和标签 #
# trainer_iter, test_iter = load_data_fashion_mnist(batch_size=32, resize=20)
# for batch_idx, (data, label) in enumerate(trainer_iter):
#     print(f"Batch {batch_idx + 1}")
#     print(f"data shape: {data.shape}, data type: {data.dtype}\nlabel shape: {label.shape}, label type: {label.dtype}")
#     break
