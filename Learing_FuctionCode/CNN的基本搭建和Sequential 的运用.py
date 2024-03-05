##Senquential 的使用
#从CIFAR10数据集来说 inputs(3,32,32)-卷积(32,32,32)-池化(32,16,16)-卷积(32,16,16)-池化(32,8,8)-卷积(64,8,8)-池化(64,4,4)-展平(64)-线性(10)
#tips:卷积层的卷积核5 图片为32 则卷积后只有 32-5+1=28 padding每加1 大小加2 maxpool 池化层池化核为2 则大小会缩小一半
import torch.nn as nn
import torch
from torch.nn import Module,Conv2d,MaxPool2d,Linear,Flatten,Sequential
from torch.utils.tensorboard import SummaryWriter
import shutil
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        # self.conv1=Conv2d(3,32,kernel_size=5,padding='same')
        # self.maxpool1=MaxPool2d(kernel_size=2)
        # self.maxpool2=MaxPool2d(kernel_size=2)
        # self.conv2=Conv2d(32,32,kernel_size=5,padding=2)
        # self.conv3=Conv2d(32,64,kernel_size=5,padding=2)
        # self.faltten=Flatten()
        # self.maxpool3=MaxPool2d(kernel_size=2)
        # self.linear1=Linear(1024,64) #64个类别
        # self.linear2=Linear(64,10)

        self.model1=Sequential(
            Conv2d(3, 32, kernel_size=5, padding='same'),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64) ,# 64个类别
            Linear(64, 10)
        )
    def forward(self,input):
        # input=self.conv1(input)
        # input=self.maxpool1(input)
        # input=self.conv2(input)
        # input=self.maxpool1(input)
        # input=self.conv3(input)
        # input=self.maxpool3(input)
        # input=self.faltten(input)
        # input=self.linear1(input)
        # output=self.linear2(input)
        output=self.model1(input)
        return output
shutil.rmtree('../logs')
writer=SummaryWriter('../logs')
tsy=Model()
input=torch.ones([64,3,32,32])
output=tsy(input)
print(output.shape)
writer.add_graph(tsy,input) #网络层的描绘
writer.close()