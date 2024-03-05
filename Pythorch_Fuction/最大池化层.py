#kernel_size – the size of the window to take a max over
#取该覆盖window下最大的一个数
#保留图像特征，减少数据数量，加快训练速度
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="datast1",train=False,transform=torchvision.transforms.ToTensor(),download=True
                                       )
dataloader = DataLoader(dataset=dataset,batch_size=64,shuffle=True,drop_last=True,num_workers=0)

'''input = torch.Tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])
input = torch.reshape(input,(-1,1,5,5))
print(input.shape)'''
class mymudule(nn.Module):
    def __init__(self):
        super(mymudule, self).__init__()
        self.max1 = torch.nn.MaxPool2d(kernel_size=3,ceil_mode=False)
    def forward(self,x):
        output = self.max1(x)
        return output
writer = SummaryWriter("logs")
mymudule1 = mymudule()
step = 0




for i in range(2):
    for data in dataloader:
        imgs, targets = data
        output = mymudule1(imgs)
        writer.add_images("epoch{}".format(i),imgs,global_step=step)
        writer.add_images("epoch1{}".format(i),output,global_step=step)
        step = step + 1

writer.close()