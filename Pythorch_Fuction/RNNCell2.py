import torch

input_size = 4
hidden_size = 4
batch_size = 1

idx2char = ['e','h','l','o']
# e-0 h-1 l-2 o-3
# 独热向量
one_hot_lookup = [[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]]
#输入序列
x_data = [1,0,2,2,3]
# 结果序列
y_data = [3,1,2,3,2]
  #如果x_data里取值1 就将one_hot_lookup 中的第一行拿走
x_one_hot = [one_hot_lookup[x] for x in x_data]
# batch_size 可理解为行   hidden_size 可理解为列
# 设置-1 可以让函数帮我们自动计算
inputs = torch.Tensor(x_one_hot).view(-1,batch_size,input_size)
print(inputs)
labels = torch.LongTensor(y_data).view(-1,1)
print(labels.shape)



class Model(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnncell = torch.nn.RNNCell(input_size=input_size,hidden_size=hidden_size)

    def forward(self,input,hidden):
        hidden = self.rnncell(input,hidden)
        return hidden
    def init_hidden(self):
        return torch.zeros(self.batch_size,self.hidden_size)


net = Model(input_size,hidden_size,batch_size)

#设置损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.1)


for epoch in range(15):
    loss = 0
    optimizer.zero_grad()
    #初始隐藏层
    hidden = net.init_hidden()
    print(hidden)
    print('预测字符串：',end='')
    for input , label in zip(inputs,labels):
        hidden = net(input,hidden)
        loss += criterion(hidden,label)
        _,idx = hidden.max(dim = 1)
        print(idx2char[idx.item()],end='')
    loss.backward()
    optimizer.step()
    print('   ,epoch [%d/15]  loss=%.4f'%(epoch+1, loss.item()))





