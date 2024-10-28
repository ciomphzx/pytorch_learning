import os

# 创建 data 目录
os.makedirs(os.path.join('..', 'data'), exist_ok = True)
# 创建 house_tiny.csv 文件
data_file = os.path.join('..', 'data', 'house_tiny.csv')
# 写入数据
with open(data_file, 'w') as f:
    f.write('NumRooms, Alley, Price\n') # 列名
    f.write('NA, Pave, 127500\n') # 每行表示一个数据样本
    f.write('2, NA, 106000\n')
    f.write('4, NA, 178100\n')
    f.write('NA, NA, 140000\n')

import pandas as pd

# 读取数据
data = pd.read_csv(data_file)
print(data)

# 处理缺失值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean()) #用均值填充缺失值
# 独热编码
inputs = pd.get_dummies(inputs, dummy_na=True)
print('input is:\n', inputs)
print('output is:\n', outputs)

import torch
# 转换为张量
input_tensor = torch.tensor(inputs.values, dtype = torch.float32)
output_tensor = torch.tensor(outputs.values, dtype = torch.float32)
print('input_tensor is:\n', input_tensor)
print('output_tensor is:\n',output_tensor)