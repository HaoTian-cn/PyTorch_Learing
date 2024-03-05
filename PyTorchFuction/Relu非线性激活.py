#非线性变换：给网络引入一些非线性的特征非线性特征越多对模型训练更好，训练出符合各种曲线和特征的模型
import torch.nn
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset1",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset=dataset,batch_size=64)


class mymodule(nn.Module):
    def __init__(self):
        super(mymodule, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self,input):
        output = self.sigmoid(input)
        return output

M_module = mymodule()

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("epoch1",imgs,global_step=step)
    output = M_module(imgs)
    writer.add_images("epoch2",output,global_step=step)
    step = step + 1
writer.close()
