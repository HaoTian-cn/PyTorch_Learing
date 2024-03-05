import torch

batch_size = 2
seq_len = 4
input_size = 3
hidden_size = 4
num_Layers = 1

cell = torch.nn.RNNCell(input_size= input_size,hidden_size= hidden_size)
cell2 = torch.nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_Layers)
# seq 表示
dataset = torch.randn(seq_len,batch_size,input_size)
hidden = torch.zeros(batch_size,hidden_size)
hidden1 = torch.zeros(num_Layers,batch_size,hidden_size)
print(hidden1.shape)
print(dataset,dataset.shape)
print('hidden1:',hidden1,hidden1.shape)

  # dataset 是输入的x   hidden1 是隐藏层输入
out , input1 = cell2(dataset,hidden1)
print(input1)
print('Input1 size :',input1.shape)
#  out 的输出是原输入序列数 + batchsize + hiddensize
print('Output size :',out.shape)

print(out)



# RNN:batch_first 将 batchsize 与 seqlen 做一个换位

