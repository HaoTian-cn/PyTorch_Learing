#搭建神经网络 十分类的网络
import torch.nn as nn
from torch.nn import  Conv2d,MaxPool2d,Flatten,Linear,Sequential
import torch
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = Sequential(
            Conv2d(3, 32, kernel_size=5, padding='same'),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),  # 64个类别
            Linear(64, 10)
        )
    def forward(self,x):
        return self.model(x)
