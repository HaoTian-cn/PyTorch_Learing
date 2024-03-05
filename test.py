import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.nn import Linear,ReLU,MSELoss,Sequential,Sigmoid,BatchNorm1d,Flatten,Softmax
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import string
#归一化函数
def guiyi(x):
    mean=np.sum(x,axis=0)
    return x/mean
#数据处理
data=pd.read_excel('data.xlsx')
# data[data.shape[2]]=np.arange(7)
xname=['年份','种子费','化肥费','农药费','机械费','灌溉费']
x=np.array(data[xname])
# x=guiyi(x)
yname=['单产']
y=np.array(data[yname])
# y_sum=np.sum(y)
# y=guiyi(y)
x=torch.tensor(x).float()
y=torch.tensor(y).float()
num=round(x.shape[0]*0.6)
train_data=zip(x[0:num],y[0:num])
test_data=zip(x[num+1:],y[num+1:])

train_data=DataLoader(list(train_data),batch_size=3,shuffle=True)
test_data=DataLoader(list(test_data),batch_size=1,shuffle=False)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model=Sequential(
            BatchNorm1d(6),
            Linear(6,10),
            Sigmoid(),
            Linear(10,5),
            Sigmoid(),
            Linear(5,1),
        )
    def forward(self,x):
        return self.model(x)
tsy=Model()
tsy=tsy.cuda()
#损失函数
loss_fun=nn.MSELoss()
loss_fun.cuda()
#优化器
op=torch.optim.Adadelta(tsy.parameters(),0.1)
#训练函数
def train_fun(epoch):
    sum_loss=0
    for x,target in train_data:
        x=x.cuda()
        target=target.cuda()
        y=tsy(x)
        loss=loss_fun(y,target)
        sum_loss+=loss
        op.zero_grad()
        loss.backward()
        op.step()
    print('第{}代训练完成 sumloss：{}'.format(epoch+1,sum_loss))
#测试函数
def test_fun(epoch):
    sum_loss=0
    with torch.no_grad():
        for x,target in test_data:
            x = x.cuda()
            target = target.cuda()
            y=tsy(x)
            loss=loss_fun(y,target)
            sum_loss+=loss
        print('第{}代测试完成 sumloss：{}'.format(epoch+1,sum_loss))

#测试
for epoch in range(100):
    tsy.train()
    train_fun(epoch)
    tsy.eval()
    test_fun(epoch)
