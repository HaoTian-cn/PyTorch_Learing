import torch.nn
import torchvision
from torch import nn
from torch.nn import Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset1",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset=dataset,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
class my_module1(nn.Module):
    def __init__(self):
        super(my_module1, self).__init__()
        '''self.conv1 = torch.nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,padding=2,stride=1)#第一次卷积
        self.MAX1 = torch.nn.MaxPool2d(kernel_size=2)#第一次最大池化
        self.conv2 = torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,padding=2,stride=1)
        self.MAX2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv3 = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,padding=2,stride=1)
        self.MAX3 = torch.nn.MaxPool2d(kernel_size=2)
        self.faltten = torch.nn.Flatten()#将其展平
        self.linear1 = torch.nn.Linear(64*4*4,64)#线性层将其展开成理想目标
        self.linear2 = torch.nn.Linear(64,10)'''
        self.model1 = Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2, stride=1),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=1),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=1),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(1024, 64),
            torch.nn.Linear(64, 10)
        )
    def forward(self,input):
        output = self.model1(input)
        return output

M_module = my_module1()
print(M_module)
input = torch.ones((64,3,32,32))#检查网络是否正确搭建
output = M_module(input)
print(output.shape)


writer = SummaryWriter("logs_seq")
writer.add_graph(M_module,input)#通过tensorboard查看网络层
writer.close()