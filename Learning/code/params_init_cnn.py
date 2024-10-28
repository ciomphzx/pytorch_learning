import torch
import torch.nn as nn
import torch.nn.functional as F
from torchstat import stat
# from pytorch_model_summary import summary
from torchinfo import summary

## 在卷积中进行自动padding的操作会额外引入其他操作, 导致模型转onnx后网络结构中有Cast、DIV、SUB等操作, 网络结构复杂 ##
class Conv2dAutoPadding(nn.Module):
    ## 根据输入特征层尺寸、卷积核尺寸、步长和输出特征层尺寸自动计算 padding 值完成卷积 ## 
    ## 默认不指定输出特征层尺寸, 此时输出特征层尺寸与输入特征层尺寸相等 ##
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, output_size=None, **kwargs):
        """
        Args:
            in_channels: 输入特征层通道数
            out_channels: 输出特征层通道数
            kernel_size: 卷积核尺寸
            stride: 卷积核滑动步长, 默认为1
            output_size: 输出特征层尺寸
                         默认None: 与输入特征层尺寸相同
                         'half': 下采样
                         (output_height, output_width): 指定输出特征层尺寸
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.output_size = output_size
    
    def forward(self, x):
        # x.size: [batch_size, channel, height, width]
        input_height, input_width = x.size(2), x.size(3)

        if self.output_size is not None:
            if self.output_size == "half":
                output_height = input_height // 2
                output_width = input_width // 2
            else:
                output_height, output_width = self.output_size
            padding_height = max((output_height - 1) * self.stride + self.kernel_size - input_height, 0)
            padding_width = max((output_width - 1) * self.stride + self.kernel_size - input_width, 0)
        else:
            padding_height = max((input_height - 1) * self.stride + self.kernel_size - input_height, 0)
            padding_width = max((input_width - 1) * self.stride + self.kernel_size - input_width, 0)

        pad_top = padding_height // 2
        pad_bottom = padding_height - pad_top
        pad_left = padding_width // 2
        pad_right = padding_width - pad_left
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return self.conv(x)

    ## 获取权重 ##
    def weight(self):
        return self.conv.weight

    ## 获取偏置 ##
    def bias(self):
        return self.conv.bias

# CNN 模型定义
class CNN_Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv2dAutoPadding(192, 128, kernel_size=1),
            # nn.LeakyReLU(0.1),
            # Conv2d(128, 256, kernel_size=3),
            # nn.LeakyReLU(0.1),
            # Conv2d(256, 256, kernel_size=1),
            # nn.LeakyReLU(0.1),
            # Conv2d(256, 512, kernel_size=3),
            # nn.LeakyReLU(0.1),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        print(x.shape)
        return x

## 随机初始化权重 ##
def init_weight_normal(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight)
        nn.init.zeros_(m.bias)

if __name__ == "__main__":
    torch.cuda.empty_cache()    # 清除显存的缓存
    # 初始化模型
    model = CNN_Module()
    # 应用随机初始化
    model.apply(init_weight_normal)

    # #　保存模型参数
    torch.save(model.state_dict(), '../model/pth/cnn_module.pth')
    print('model params saved done!')

    # # 统计模型网络结构信息
    # summary(model, input_size=(1, 3, 448, 448))