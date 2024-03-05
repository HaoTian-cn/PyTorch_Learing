import torch
batch_size=1
seq_size=5
input_size=4
hidden_size=4
num_layer=1
idx2char=['e','h','l','o'] #字典
x_data=[1,0,2,2,3] #hello
y_data=[3,1,2,3,2] #ohlol
one_hot_lookup=[[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]] #读入向量
x_one_hot=[one_hot_lookup[x] for x in x_data]
inputs=torch.Tensor(x_one_hot).reshape(-1,batch_size,input_size) #-1其实就是seq_size
# print(inputs)
labels=torch.LongTensor(y_data)
#模型
class Model(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,num_layer):
        super(Model, self,).__init__()
        self.batch_size=batch_size
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layer=num_layer
        self.rnn=torch.nn.RNN(input_size=self.input_size,hidden_size=hidden_size,num_layers=num_layer)
    def forward(self,input):
        hidden=torch.zeros(self.num_layer,self.batch_size,self.hidden_size)
        out,_=self.rnn(input,hidden) #out是每个输出的结果
        return out.reshape(-1,self.hidden_size)
    def init_hidden(self):
        return torch.zeros(self.batch_size,self.hidden_size)
tsy=Model(input_size,hidden_size,batch_size,num_layer)

#损失函数
loss_fun=torch.nn.CrossEntropyLoss()

#优化器
op=torch.optim.Adam(tsy.parameters(),0.1)
#训练过程
for epoch in range(15):

    output=tsy(inputs)

    loss=loss_fun(output,labels)
    idx=output.argmax(1)
    op.zero_grad()
    loss.backward()
    op.step()
    print(''.join([idx2char[x] for x in idx])) #join函数将列表中的各个字符串联起来
