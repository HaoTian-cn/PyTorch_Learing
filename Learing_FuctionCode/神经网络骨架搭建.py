#CNN神经网络的骨架 输入-卷积-非线性-卷积-非线性-输出
import torch.nn as nn
import torch
class Model(nn.Module):
    def __init__(self):
        super().__init__() #调用父类属性
    def forward(self,imput):
        output=imput+1
        return output
tsy=Model()
x=torch.tensor(1)
print(tsy(1))
# cov2d的使用 cov2d(in_channels,out_channels,kernel_size,stride=1,paddng=0)
#in_channels输入图像的通道数 out_channels输出图像的通道数 kernel_size可以是一个元组或者数字 定义卷积核大小
#stride 是步长默认为1 padding 是否填充边缘默认为0，没有填充
#卷积核是自动生成，按照一定分布随机采样
