#relu 输入不限制，sigmod是常用的激活函数
#relu 或者sigmod 中的Inplace Relu(input，inplace=True) 是否对原来变量进行结果变换 inplace=false可以对原始数据保留
import torch.nn as nn
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.nn import MaxPool2d,Conv2d,Sigmoid
import shutil #删除文件夹的模块
from torch.utils.tensorboard import SummaryWriter
shutil.rmtree('../logs')
dataset=torchvision.datasets.CIFAR10('../dataset',train=False,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64) #将64张图合成一张图
wrietr=SummaryWriter('../logs')
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
        self.maxpool=MaxPool2d(kernel_size=3,stride=1,padding=0) #**
        self.sigmoid=Sigmoid()
    def forward(self,x):
        # x=self.conv1(x) 卷积
        x=self.maxpool(x) #** 池化
        x=self.sigmoid(x)
        return x
tsy=Model()
step=0
for data in dataloader:
    imgs,target=data
    output=tsy(imgs)
    print(imgs.shape)
    print(output.shape)
    wrietr.add_images('input',imgs,step)
    output=torch.reshape(output,(-1,3,30,30))
    wrietr.add_images('output', output, step)
    step +=1