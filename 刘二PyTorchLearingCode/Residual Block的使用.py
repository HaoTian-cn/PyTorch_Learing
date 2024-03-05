#全链接的神经网络处理minist数据集
import torch
from torch.utils.data import DataLoader,Dataset
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.nn import Linear,Sequential,CrossEntropyLoss,Flatten,ReLU,Softmax,Conv2d,MaxPool2d
#minist数据集
tran=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train=torchvision.datasets.MNIST('../dataset2',download=True,train=True,transform=tran)
test=torchvision.datasets.MNIST('../dataset2',download=True,train=False,transform=tran)

train_data=DataLoader(train,batch_size=64,shuffle=True)
test_data=DataLoader(test,batch_size=64)
#模型
class ResidualBlock(nn.Module): #残差网络跳链接，防止梯度消失，H（x）=x+F（x） 使得梯度大于1 避免小于1累乘趋近与0
    def __init__(self,channels):
        super(ResidualBlock, self).__init__()
        self.Res=Sequential(
            Conv2d(channels,channels,3,padding=1),
            ReLU(),
            Conv2d(channels,channels,3,padding=1),
        )#先不激活F()
        self.relu=ReLU()
    def forward(self,input):
        y=self.Res(input)
        return self.relu(input+y) #**最后一步在输出和输入相加再激活 Relu(F(x)+x )
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model=Sequential(
            Conv2d(1, 10, 5),
            ReLU(),
            MaxPool2d(2),
            ResidualBlock(channels=10),#*
            Conv2d(10,20,5),
            ReLU(),
            MaxPool2d(2),
            ResidualBlock(channels=20),#*
            Flatten(), #因为有batch_size所以需要平铺图像
            Linear(4*4*20,10)
        )
    def forward(self,x):
        return self.model(x)
tsy=Model()
tsy=tsy.cuda()
#损失函数
loss_fun=CrossEntropyLoss()
loss_fun=loss_fun.cuda()
#优化器
op=torch.optim.SGD(tsy.parameters(),lr=0.01,momentum=0.9)
#训练过程
def train_progress(epoch):
    sum_loss=0
    for img,target in train_data:
        img=img.cuda()
        target=target.cuda()
        output=tsy(img)
        loss=loss_fun(output,target)
        op.zero_grad()
        loss.backward()
        op.step()
        sum_loss+=loss
    print('第{}轮训练完成,其总loss为sum_loss={}'.format(epoch+1,sum_loss))
#测试过程
def test_progress(epoch):
    rote=0
    sum_loss=0
    with torch.no_grad():
        for img, target in test_data:
            img = img.cuda()
            target = target.cuda()
            output = tsy(img)
            loss = loss_fun(output, target)
            sum_loss += loss
            real=(output.argmax(1)==target).sum()
            rote+=real
        print('第{}轮测试完成,其总loss为sum_loss={}'.format(epoch+1,sum_loss))
        print('正确率为{}%'.format(rote/len(test)*100))
for epoch in range(20):
    tsy.train()
    train_progress(epoch)
    tsy.eval()
    test_progress(epoch)
