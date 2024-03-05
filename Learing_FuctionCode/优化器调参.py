#优化器的损失函数一个backward()反向传播对应参数的梯度 使用优化器对梯度进行调整 达到减少损失函数的目的
#1.构造优化器 （模型参数，学习速率，...） 2.优化器的step 方法 利用梯度对参数进行更新optimizer.zero_grad()** ,optimizer.step()
#优化器的算法 torch.optim.Adadelta(参数（params），学习速率（LR），...（基于算法的设置）)
import torch.nn as nn
import torchvision
import torch
from torch.nn import Module,Conv2d,MaxPool2d,Linear,Flatten,Sequential,CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
import shutil

from torch.utils.data import DataLoader
dataset=torchvision.datasets.CIFAR10('../dataset',train=False,transform=torchvision.transforms.ToTensor())
data=DataLoader(dataset,1)
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()

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

        output=self.model1(input)
        return output
tsy=Model()
optim =torch.optim.SGD(tsy.parameters(),lr=0.01) #tsy.parameters是模型参数权重 ，lr是学习速率 **
for i in range(20): #外层嵌套的这个循环是将模型训练的更加精准
    sum_loss=0.0 #小数形式
    for img,target in data:
        outputs=tsy(img)
        my_loss=CrossEntropyLoss()
        loss=my_loss(outputs,target)
        optim.zero_grad() #梯度归0**
        loss.backward() #**
        optim.step() #对每个参数进行调优**
        sum_loss=sum_loss+loss
    print(sum_loss)