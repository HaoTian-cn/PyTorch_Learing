#如果加入正则化层可以加速训练速度 用的不是很多
#线性层的使用
#dropout层防止过拟合
#该项主要是线性层的应用 Linear(in_feature,out_feature)
import torch.nn as nn
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.nn import MaxPool2d,Conv2d,Linear
import shutil #删除文件夹的模块
from torch.utils.tensorboard import SummaryWriter
shutil.rmtree('../logs')
dataset=torchvision.datasets.CIFAR10('../dataset',train=False,download=True,transform=torchvision.transforms.ToTensor())
writer=SummaryWriter('../logs')
dataloader=DataLoader(dataset,batch_size=64)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear=Linear(196608,10)
    def forward(self,x):
        output=self.linear(x)
        return output
tsy=Model()
step=0
for img,target in dataloader:
    print(img.shape)
    print(torch.reshape(img,[1,1,1,-1]).shape)
    new_img=torch.flatten(img) #将矩阵转为向量
    out=tsy(new_img)
    print(out.shape)
    step+=1
