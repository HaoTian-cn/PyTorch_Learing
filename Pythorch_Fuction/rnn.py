import torch

input_size = 4
hidden_size = 4
batch_size = 1
num_Layers = 1
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
# batch_size 可理解为行   input_size 可理解为列
# 设置-1 可以让函数帮我们自动计算序列数seqlen
inputs = torch.Tensor(x_one_hot).view(-1,batch_size,input_size)
labels = torch.LongTensor(y_data)  #变为整型tensor类型
print(inputs.shape)
print(labels,inputs)

class Model(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,num_Layers):
        super(Model, self).__init__()
        self.num_Layers = num_Layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnncell = torch.nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_Layers)

    def forward(self,input):
        hidden = torch.zeros(self.num_Layers,self.batch_size, self.hidden_size)   #h0 初始隐藏层
        out, _ = self.rnncell(input,hidden)
        return out.view(-1,self.hidden_size)


net = Model(input_size,hidden_size,batch_size,num_Layers)
print()
#设置损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.1)


for epoch in range(15):
    loss = 0
    optimizer.zero_grad()
    #初始隐藏层
    output  = net(inputs)
    loss += criterion(output,labels)
    loss.backward()
    optimizer.step()
    print(output)
    _,idx = output.max(dim=1)  #返回每一行的最大值对应的下标
    print(idx)
    idx = idx.data.numpy()
    print(idx)#将idx变为numpy向量
    print('Predicted:',''.join(idx2char[x] for x in idx),end='') #输出idx对应下标在idx2char里面的字符
    print('   ,epoch [%d/15]  loss=%.3f'%(epoch+1, loss.item()))