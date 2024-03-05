#其实就是将复杂网络重复的块变成一个类然后里面拼接调用
import torch.nn as nn
from torch.nn import Sequential,Conv2d,AvgPool2d
import torch
class Inception(nn.Module):
    def __init__(self,in_channels): #输入的张量通道数
        super(Inception, self).__init__()
        self.inception1=Sequential(
            AvgPool2d(3,padding=1),
            Conv2d(in_channels,24,1),
        )
        self.inception2=Conv2d(in_channels,16,1)
        self.inception3=Sequential(
            Conv2d(in_channels,16,1),
            Conv2d(16,24,5,padding=2)
        )
        self.inception4=Sequential(
            Conv2d(in_channels,16,1),
            Conv2d(16,24,3,padding=1),
            Conv2d(24,24,5,padding=2)
        )
    def forward(self,x):
        x1=self.inception1(x)
        x2=self.inception2(x)
        x3=self.inception3(x)
        x4=self.inception4(x)
        output=torch.cat([x1,x2,x3,x4],dim=1)
        return output