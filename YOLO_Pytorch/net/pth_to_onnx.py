## pth转onnx文件 ##
from yolo import YOLOv1
import torch

if __name__ == '__main__':
    # 清除缓存
    torch.cuda.empty_cache()
    # 获取GPU设备
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    
    # 创建模型实例对象
    net = YOLOv1()
    
    # pth文件路径
    pth_path = '../model/pth/YOLOv1.pth'
    # 加载pth文件模型参数
    static_dict = torch.load(pth_path)
    net.load_state_dict(static_dict)
    print('load moadel params done!')
    
    # 模型移到GPU
    net = net.to(device)
    
    # 创建输入
    input = torch.randn(1, 3, 192, 192).to(device)
    
    # onnx文件路径
    onnx_path = '../model/onnx/YOLOv1.onnx'

    # 导出模型为 ONNX 格式
    torch.onnx.export(
        net,  # 要导出的模型
        input,  # 模型的输入张量
        onnx_path,  # 输出的 ONNX 文件路径
        input_names=['input'],  # 输入张量的名字
        output_names=['output'],  # 输出张量的名字
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},  # 支持动态批量大小
        opset_version=12  # 指定 opset 版本
    )