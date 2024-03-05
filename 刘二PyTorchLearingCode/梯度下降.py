import numpy as np
import matplotlib.pyplot as plt
x=[1.0,2.0,3.0]
y=[2.0,4.0,6.0]
w=1 #初始化权值
def forward(x):
    return x*w
def cost(xs,ys): #求损失函数
    cost=0
    for x,y in zip(xs,ys):
        y_pred=forward(x)
        cost+=(y-y_pred)**2
    return  cost/len(xs)
def grad(xs,ys): #求梯度,整个样本的梯度
    grad=0
    for x,y in zip(xs,ys):
        grad+=2*x*(x*w-y)
    return grad/len(xs)
'''
随机梯度下降直接return 2*x*(x*w-y)就行
'''
loss_list=[]
for epoch in range(100):

    cost_val=cost(x,y)
    loss_list.append(cost_val)
    grad_val=grad(x,y)
    w-=0.01*grad_val #0.01为学习率
    print('EPOCH:',epoch,'W:',w,'LOSS:',cost_val)
plt.plot(range(100),loss_list)
plt.xlabel('epoch')
plt.ylabel('LOSS')
plt.show()