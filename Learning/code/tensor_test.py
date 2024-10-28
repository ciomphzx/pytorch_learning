#pytorch的基础知识
import torch

x = torch.arange(12)
print("x is:\n",x)
#tensor的维度:
print("the dim of x is:\n", x.shape)
#tensor的元素个数,维度是1的情况下元素个数=维度
print("the element nums in x is:\n", x.numel())

#调整tensor的维度(n行m列矩阵,自动计算行数3行4列)
x_resize = x.reshape(-1, 4)
print("x_resize is:\n", x_resize)
print("the dim of x_resize is:\n", x_resize.shape)
print("the element nums in x_resize is:\n", x_resize.numel())

#随机初始化值的tensor:元素服从均值为0，标准差为1的标准高斯分布
x_rand = torch.rand(3, 4)
print("x_rand is:\n", x_rand)

#所有元素都为0的tensor,dtype参数指定数据类型
x_zero = torch.zeros(2, 3, 4, dtype = torch.float32)
print("x_zero is:\n", x_zero)
print("the dim of x_zero is:\n", x_zero.shape)
print("the elements in x_zero is:\n", x_zero.numel())

#所有元素都为1的tensor
x_one = torch.ones(2, 3, 4, dtype = torch.int8)
print("x_one is:\n", x_one)
print("the dim of x_one is:\n", x_one.shape)
print("the element nums of x_one is:\n", x_one.numel())

#初始化tensor并赋值
tensor_x = torch.tensor([[1, 2, 3],
                         [4, 5, 6]])
tensor_y = torch.tensor([[1, 1, 1],
                         [2, 2, 2]])
print("tnesor_x is:\n", tensor_x)
print("tensor_y is:\n", tensor_y)
#tensor的运算，逐元素运算
tensor_sum = tensor_x + tensor_y
print("tensor_x + tensor_y is:\n", tensor_sum)
tensor_sub = tensor_x - tensor_y
print("tensor_x - tensor_y is:\n", tensor_sub)
tensor_mul = tensor_x * tensor_y
print("tensor_x * tensor_y is:\n", tensor_mul)
tensor_div = tensor_x / tensor_y
print("tensor_x / tensor_y is:\n", tensor_div)
tensor_pow = tensor_x ** tensor_y
print("(tensor_x)^(tensor_y) is:\n", tensor_pow)

#tensor concatenate
tensor_concat_row = torch.cat((tensor_x, tensor_y), dim = 0)
print("tensor_concat_row is:", tensor_concat_row)
print("the dim of tensor_concat_row is:", tensor_concat_row.shape)
tensor_concat_col = torch.cat((tensor_x, tensor_y), dim = 1)
print("tensor_concat_col is:", tensor_concat_col)
print("the dim of tensor_concat_col is:", tensor_concat_col.shape)

#tensor元素索引
tensor_x = torch.arange(20).reshape(4, -1)
tensor_index = tensor_x[0:3, :]
print("tensor_index is:", tensor_index)