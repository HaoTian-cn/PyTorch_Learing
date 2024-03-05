import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


#核函数
def kernel(ker,x1,x2,gamma):
    '''

    :param ker:核函数类型
    :param x1: 输入数据
    :param x2: 输入数据
    :param gamma:rbf核带宽
    :return:
    '''
    k=None
    if not ker or ker == 'primal':
        k = x1
    elif ker == 'linear':
        if x1 is not None:
            k = sklearn.metrics.pairwise.linear_kernel(
                np.asarray(x1).T, np.asarray(x2).T)
        else:
            k = sklearn.metrics.pairwise.linear_kernel(np.asarray(x1).T)
    elif ker == 'rbf':
        if x2 is not None:
            k = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(x1).T, np.asarray(x2).T, gamma)
        else:
            k = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(x1).T, None, gamma)
    return k

class TCA:
    def __init__(self,kernel_type='primal',dim=30,lamb=1,gamma=1):
        '''

        :param kernel_type:核函数类型
        :param dim: 数据的总条数一般是源域和目标域的两倍
        :param lamb:等式中的λ值
        :param gamma:rbf核带宽
        '''
        self.kernel_type=kernel_type
        self.dim=dim
        self.lamb=lamb
        self.gamma=gamma
    def fit(self,xs,xt):
        '''

        :param xs:源域 数据
        :param xt: 目标域数据
        :return: 经过TCA变换后源域和目标域的新数据
        '''
        x=np.hstack((xs.T,xt.T)) #水平叠加
        x /= np.linalg.norm(x, axis=0) #求矩阵的范数 ,x(1,i)/sqrt(x(1,i)^2+x(2,i)^2
        m,n=x.shape
        ns,nt=len(xs),len(xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1)))) #源域和目标域的1，-1赋值
        m=e*e.T #矩阵
        m = m / np.linalg.norm(m, 'fro') #求范数矩阵的平方和再开方
        h = np.eye(n) - 1 / n * np.ones((n, n))
        k = kernel(self.kernel_type, x, None, gamma=self.gamma) #核函数

        return h
if __name__=='__main__':
    tsy=TCA()
    a=np.array([[1,2,3]],dtype=float).reshape(3,1)
    b=np.array([[4,5,6]],dtype=float).reshape(3,1)

    print(tsy.fit(a,b))