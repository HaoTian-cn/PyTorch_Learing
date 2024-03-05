# cov2d的使用 cov2d(in_channels,out_channels,kernel_size,stride=1,paddng=0)
#in_channels输入图像的通道数 out_channels输出图像的通道数 kernel_size可以是一个元组或者数字 定义卷积核大小
#stride 是步长默认为1 padding 是否填充边缘默认为0，没有填充
#卷积核是自动生成，按照一定分布随机采样
import torch.nn as nn
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.nn import Conv2d
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

    def forward(self,x):
        x=self.conv1(x)
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