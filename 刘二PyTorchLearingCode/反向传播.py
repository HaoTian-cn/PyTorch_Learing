#back propagation
import torch
import numpy as np
import matplotlib.pyplot as plt
x=[1.0,2.0,3.0]
y=[2.0,4.0,6.0]
w=torch.tensor([1.0])
w.requires_grad=True
def forward(x):
    return x*w
def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)**2
for epoch in range(100):
    for xs,ys in zip(x,y):
        l=loss(xs,ys)
        l.backward()
        print('\tgrad:',x,y,w.item())
        w.data=w.data-0.01*w.grad.data #防止产生计算图 否则吃内存
        w.grad.data.zero_()
    print('progress',epoch,l.item())