import torch.nn
from torch import nn

#  搭建神经网络模型
class mymodule(nn.Module):
    def __init__(self):
        super(mymodule, self).__init__()
        self.seq_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(64*4*4,64),
            torch.nn.Linear(64,3)
        )
    def forward(self,input):
        output = self.seq_1(input)
        return output

# 测试网络正确性
if __name__ == '__main__':
    my_demo = mymodule()
    input = torch.ones((64,3,32,32))
    output = my_demo(input)
    print(output.shape)


