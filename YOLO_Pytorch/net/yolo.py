## 定义 YOLO 网络结构 ##
import torch
import torch.nn as nn
from torchinfo import summary

class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),  # Conv 1
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max Pooling 1
            
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),  # Conv 2
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max Pooling 2
            
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0),  # Conv 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # Conv 4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),   # Conv 5
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),   # Conv 6
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max Pooling 3
            
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),  # Conv 7
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # Conv 8
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),  # Conv 9
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # Conv 10
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),  # Conv 11
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # Conv 12
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),  # Conv 13
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # Conv 14
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),   # Conv 15
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),  # Conv 16
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max Pooling 4
            
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),  # Conv 17
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),  # Conv 18
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),  # Conv 19
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),  # Conv 20
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1), # Conv 21
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1), # Conv 22

            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1), # Conv 23
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)  # Conv 24
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 3 * 3, 4096),  # FC 1
            nn.Linear(4096, 1470)  # FC 2
        )

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x

# 只有运行这个文件时, 后面的测试代码才会执行
if __name__ == "__main__":
    model = YOLOv1()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # 测试输入
    dummy_input = torch.randn(1, 3, 192, 192).to(device)
    # 输出
    output = model(dummy_input)
    print("Output shape:", output.shape)  # 应该是 (1, 1470)
    # 统计模型网络结构信息
    summary(model, input_size=(1, 3, 192, 192))
