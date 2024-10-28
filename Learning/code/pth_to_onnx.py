import onnx
import torch
from torch import nn
from params_init_cnn import CNN_Module

# 创建 YOLOv1 模型实例
model = CNN_Module()

# 加载权重
loaded_state_dict = torch.load("../model/pth/cnn_module.pth")

# 尝试加载新修正的状态字典
model.load_state_dict(loaded_state_dict)

# 将模型的模式设置为评估模式
model.eval()

# 创建一个虚拟的输入张量
dummy_input = torch.randn(1, 3, 448, 448)  # 批量大小为1，输入通道为3，图像大小为448x448

# 指定 ONNX 文件的保存路径
onnx_file_path = "../model/onnx/cnn_module.onnx"

# 导出模型为 ONNX 格式
torch.onnx.export(
    model,  # 要导出的模型
    dummy_input,  # 模型的输入张量
    onnx_file_path,  # 输出的 ONNX 文件路径
    input_names=['input'],  # 输入张量的名字
    output_names=['output'],  # 输出张量的名字
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},  # 支持动态批量大小
    opset_version=12  # 指定 opset 版本
)

print("onnx saved done!")
