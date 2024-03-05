
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
        self.linear = torch.nn.Linear(196608,10)
    def forward(self,input):
        output = self.linear(input)
        return output

M_module = mymodule()
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    #torch.reshape(imgs,(1,1,1,-1))
    output = torch.flatten(imgs)
    print(output.shape)
    output = M_module(output)
    print(output.shape)


