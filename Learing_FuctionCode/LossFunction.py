#一方面计算实际输出和目标之间的差距，另一方面为我们更新输出提供了一定的依据
#反向传播，L1loss(input,target) 用的挺多的MSEloss
#交叉熵 对分类问题 有n个类别
#对卷积核进行调整 优化
import torch
import torch.nn as nn
from torch.nn import Conv2d,MaxPool2d,Linear,MSELoss,CrossEntropyLoss #交叉熵
input=torch.tensor([1,2,3],dtype=torch.float32)
target=torch.tensor([4,5,6],dtype=torch.float32)
loss=MSELoss()
mse_loss=loss(input,target)
print(mse_loss)

loss_x=torch.tensor([0.1,0.2,0.3]) #每个项的概率
print(loss_x)
print(loss_x.shape)
loss_y=torch.tensor([1]) #target
loss_x=torch.reshape(loss_x,[1,3]) #reshape x为 1,3的 否则会报错
print(loss_x)
loss_cross=CrossEntropyLoss()
print(loss_x.shape)
out=loss_cross(loss_x,loss_y)
print(loss_cross(loss_x,loss_y))
out.backward() #反向传播形成梯度 **


