#应用方式 去pytorch官网上查找模型 ，相当于已经搭建好的框架可以直接拿来使用
import torchvision
import torch.nn as nn
vgg16=torchvision.models.vgg16()
# print(vgg16)
# train_data=torchvision.datasets.CIFAR10('../dataset',train=False,transform=torchvision.transforms.ToTensor())
#修改vgg16模型
vgg16.add_module('add_linear',nn.Linear(1000,10)) #在最后加一层线性层
vgg16.classifier[6]=nn.Linear(4096,10) #修改classifier某一层
print(vgg16)