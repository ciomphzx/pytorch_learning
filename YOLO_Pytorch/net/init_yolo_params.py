## 初始化 YOLO 网络参数 ##
import torch
import torch.nn as nn
from yolo import YOLOv1

def init_weight(m):
    if isinstance(m, nn.Conv2d):    # 卷积初始化
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):  # 线性层初始化
        nn.init.normal_(m.weight)
        nn.init.normal_(m.bias)

if __name__ == '__main__':
    model = YOLOv1()
    model.apply(init_weight)
    # 保存模型参数
    # torch.save(model.state_dict(), '../model/pth/YOLOv1.pth')
    # print('model saved done!')
    torch.save(model.model.state_dict(), '../model/pth/YOLOv1_conv_layers_one.pth')
    print('model conv layers saved done!')