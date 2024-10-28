## 加载预训练权重: 只加载特征提取的卷积层的网络参数 ##
from yolo import YOLOv1
import torch
import onnx
import collections

if __name__ == '__main__':
    # 获取GPU设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 实例化模型对象
    model = YOLOv1()
    # 获取模型的当前状态字典
    model_state_dict = model.state_dict()
    
    # 卷积层参数(即预训练模型)
    pretrained_state_dict = torch.load('../model/pth/YOLOv1_conv_layers_one.pth')
    # pretrained_state_dict = torch.load('./YOLO_Pytorch/model/pth/YOLOv1_conv_layers_one.pth')    #   debug时的相对路径设置
    
    ## 只更新预训练模型中存在的层的权重 ##
    # 调整键名称并构造新的 state_dict
    new_state_dict = collections.OrderedDict()
    for k, v in pretrained_state_dict.items():
        new_key = f'model.{k}' if 'model.' not in k else k  # 添加 'model.' 前缀
        if new_key in model_state_dict:
            new_state_dict[new_key] = v

    # 加载新的 state_dict
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict, strict=False)
    print('load pretrained model done!')

    #　模型转移到GPU
    model = model.to(device)

    ## 保存onnx模型, 验证是否成功加载了卷积层的参数 ##
    # 创建输入
    input = torch.randn(1, 3, 192, 192).to(device)
    
    # onnx文件路径
    onnx_path = '../model/onnx/YOLOv1_saved.onnx'

    # 导出模型为 ONNX 格式
    torch.onnx.export(
        model,  # 要导出的模型
        input,  # 模型的输入张量
        onnx_path,  # 输出的 ONNX 文件路径
        input_names=['input'],  # 输入张量的名字
        output_names=['output'],  # 输出张量的名字
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},  # 支持动态批量大小
        opset_version=12  # 指定 opset 版本
    )
    print('model params saved done!')
