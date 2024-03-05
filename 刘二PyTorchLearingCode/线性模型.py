import numpy as np
import matplotlib.pyplot as plt
x=[1.0,2.0,3.0]
y=[2.0,4.0,6.0]

def forward(x):
    return x*w
def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)**2
w_list=[]
mse_list=[]
for w in np.arange(0,4.1,0.1):
    print('w=',w)
    l_sum=0
    for x_hat ,y_hat in zip(x,y):
        y_pred=forward(x_hat)
        loss_1=loss(x_hat,y_hat)
        l_sum+=loss_1
        print('\t',x_hat,y_hat,loss_1)
    print('MSE=',l_sum/len(x))
    w_list.append(w)
    mse_list.append(l_sum/len(x))
#画图
plt.plot(w_list,mse_list)
plt.xlabel('W')
plt.ylabel('LOSS')
plt.show()