#### 此文件是torch中张量的计算以及计算后维度的变化 ####
import torch

# 创建标量
scaler_x = torch.tensor(1.0)
scaler_y = torch.tensor(2.0)

print('scaler_x:', scaler_x)
print('scaler_y:', scaler_y)
print('scaler_x + scaler_y:', scaler_x + scaler_y)
print('scaler_x - scaler_y:', scaler_x - scaler_y)
print('scaler_x * scaler_y:', scaler_x * scaler_y)
print('scaler_x / scaler_y:', scaler_x / scaler_y)

# 创建向量
vectot_x = torch.arange(1, 4)
vectot_y = torch.arange(3, 6)

print('vectot_x:', vectot_x)
print('vector_x has element number:', vectot_x.numel())
print('vector_x has length:', len(vectot_x))

print('vector_x has size:', vectot_x.size())
print('vxctor_x has shape:', vectot_x.shape)
#向量的维度是张量的轴数
print('vxctor_x has dimension:', vectot_x.dim())
print('vector_x[0]', vectot_x[0])
print('vectot_y:', vectot_y)
print('vectot_x + vectot_y:', vectot_x + vectot_y)
print('vectot_x - vectot_y:', vectot_x - vectot_y)
print('vectot_x * vectot_y:', vectot_x * vectot_y)
print('vectot_x / vectot_y:', vectot_x / vectot_y)

# 创建矩阵
matrix_A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
sum_A = matrix_A.sum(axis=0, keepdim=True)
print('sum_A:\n', sum_A)
print(sum_A.size())
print('matrix_A / sum_A:\n', matrix_A / sum_A)