import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm #进度条模块
import tushare as ts
from copy import deepcopy as copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

#从tushare中获取数据
class Getdata:
    def __init__(self,stock_id,save_path):
        self.stock_id=stock_id
        self.save_path=save_path
        self.data=None
    def getdata(self):
        self.data=ts.get_hist_data(self.stock_id).iloc[::-1]
        self.data=self.data[['open','close','high','low','volume']] #特征值开盘价、收盘价、最高价、最低价、成交量这五个特征
        self.close_min=self.data['close'].min() #收盘最低价
        self.close_max=self.data['close'].max() #收盘最高价
        self.data = self.data.apply(lambda x: (x - min(x)) / (max(x) - min(x))) #将data数据归一化
        self.data.to_csv(self.save_path)#将归一化后数据载入到目录下的save_path
    def process_data(self,n):
        if self.data is None:
            self.getdata() #调用类函数
        feature=[
            self.data.iloc[i:i+n].values.tolist() for i in range(len(self.data)-n+2) if i+n<len(self.data)
        ] #n是天数 每n天制作一个小样本就是seq_size
        label=[
            self.data.close.values[i+n] for i in range(len(self.data)-n+2) if i+n<len(self.data)
        ] #n是天数
        train_x = feature[:500] #五百个样本
        test_x = feature[500:]
        train_y = label[:500]
        test_y = label[500:]
        return train_x,test_x,train_y,test_y
class Lstm(torch.nn.Module):
    def __init__(self,n):
        super(Lstm, self).__init__()
        self.lstm_layer=nn.LSTM(input_size=5,hidden_size=128,batch_first=True) #LSTM层 将输入最后输出成128个hidden out
        self.linear_layer=nn.Linear(128,1,bias=True) #线形层
    def forward(self,x):
        out1,(h_n,h_c)=self.lstm_layer(x) #得到各个输出 hidden层最后的输出为h_n
        a,b,c=h_n.shape
        out2=self.linear_layer(h_n.reshape(a*b,c)) #重构成（-1，c）
        return out2
#训练模型
def train_model(epoch,train_dataloader,test_dataloader):
    best_model=None #最优模型
    train_loss=0 #训练损失
    test_loss=0 #测试损失
    best_loss=100 #最优损失
    epoch_cnt=0 #记录循环次数
    for _ in range(epoch): #循环
        total_train_loss = 0 #本次循环的训练总损失
        total_train_num = 0 #训练次数
        total_test_loss = 0 #本次循环的测试总损失
        total_test_num = 0 #测试次数
        for x,y in tqdm(train_dataloader,
                        desc='Epoch:{} |train_loss:{} |test_loss:{}'.format(_,train_loss,test_loss)):
            #x,y是载入train_dataloader中的数据其余的只是进度条显示
            xnum=len(x)
            p=net(x) #得到网络输出
            loss=loss_fun(p,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss+=loss.item()
            total_train_num+=xnum
        train_loss=total_train_loss/total_train_num
        #测试过程中的数据仍然在改变梯度 和优化器
        for x,y in test_dataloader:
            xnum=len(x)
            p=net(x)
            loss=loss_fun(p,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_test_loss+=loss.item()
            total_test_num+=xnum
        test_loss=total_test_loss/total_test_num
        #记录模型
        # early_stop=5
        if best_loss>test_loss:
            best_loss=test_loss
            best_model=copy(net)
            epoch_cnt=0
        else:
            epoch_cnt+=1
        # 后续的early_stop次模型都没有更好的就保存模型退出循环
        if epoch_cnt>early_stop:
            torch.save(best_model.state_dict(), './lstm.pth')
            break


#测试模型
def test_model(test_dataloader_):
    pred=[] #预测输出
    label=[] #标签
    model=Lstm(5)
    model.load_state_dict(torch.load('./lstm.pth')) #载入模型参数
    model.eval() #测试模式打开
    total_test_loss = 0 #测试损失
    total_test_num = 0 #测试次数
    for x,y in test_dataloader_:
        xnum=len(x)
        p=model(x)
        loss=loss_fun(p,y)
        total_test_loss+=loss.item()
        total_test_num+=xnum
        pred.extend(p.data.squeeze(1).tolist()) #去维度输出
        label.extend(y.tolist())
    test_loss =total_test_loss/total_test_num
    return pred,label,test_loss
def plot_img(data,pred): #可视化效果
    plt.plot(range(len(pred)),pred,color='green') #预测曲线

    plt.plot(range(len(data)),data,color='b') #真实曲线
    # for i in range(0,len(pred-3),5):
    #     price=[data[i]+pred[j]-pred[i] for j in range(i,i+3)]
    #     plt.plot(range(i,i+3),price,color='r')
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close Price',fontsize=18)
    plt.show()
if __name__=='__main__':
    day_num=15
    epoch=20
    fea=5 #特征数目
    batch_size=20
    early_stop=10
    #模型
    net=Lstm(fea)
    #数据处理
    GD=Getdata(stock_id='000963',save_path='./data.csv')
    x_train,x_test,y_train,y_test=GD.process_data(day_num)
    #转换数据格式
    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()
    y_train = torch.tensor(y_train).float()
    print(y_train.shape)
    # print(y_train)
    y_test = torch.tensor(y_test).float()
    train_data = TensorDataset(x_train, y_train) #包装张量数据集
    train_dataLoader = DataLoader(train_data, batch_size=batch_size) #批量
    test_data = TensorDataset(x_test, y_test)
    test_dataLoader = DataLoader(test_data, batch_size=batch_size)

    #损失和优化
    loss_fun=nn.MSELoss()
    optimizer=optim.Adam(net.parameters(),lr=0.001)
    train_model(epoch,train_dataLoader,test_dataLoader)
    p,y,test_loss=test_model(test_dataLoader)
    # 画图
    pred = [ele * (GD.close_max - GD.close_min) + GD.close_min for ele in p] #将归一化返回
    # print(len(pred))
    data = [ele * (GD.close_max - GD.close_min) + GD.close_min for ele in y]
    plot_img(data, pred)

    print(test_loss)
