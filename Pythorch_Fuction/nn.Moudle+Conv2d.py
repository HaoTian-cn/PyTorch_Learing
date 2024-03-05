import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset1",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset=dataset,batch_size=64)
class MyMoudle(nn.Module):
    def __init__(self):
        super(MyMoudle, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        # 先进行relu 非线性  再使用一次卷积操作
        x = self.conv1(x)
        return x

writer = SummaryWriter("logs")
m = MyMoudle()
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("test1",imgs,global_step=step)
    imgs1 = m(imgs)
    print(imgs.shape)
    print(imgs1.shape)
    imgs1 = torch.reshape(imgs1,(-1,3,30,30))
    writer.add_images("test2", imgs1, global_step=step)
    step = step + 1

writer.close()
print(m)
#  Conv2d参数： in_channels,输入通道数，out_channels,输出通道数
#  kernel_size,卷积核的大小,stride卷积的行走步幅大小，padding，填充