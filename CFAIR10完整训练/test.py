import  torchvision
from PIL import Image

test_dataset = torchvision.datasets.CIFAR10("test_data1", train=False, transform=torchvision.transforms.ToTensor(), download=True,)
image_path = "../dataset/img_1.png"
image = Image.open(image_path)
print(image)
transform = torchvision.transforms.Compose((
    torchvision.transforms.Resize((32,32)),
    torchvision.transforms.ToTensor()
))
input = transform(image)
print(input.shape)

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
            torch.nn.Linear(64,10)
        )
    def forward(self,input):
        output = self.seq_1(input)
        return output
    pass
model = torch.load("module_19.pth")
print(model)

input = torch.reshape(input,(1,3,32,32))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input = input.to(device)

with torch.no_grad():
    output = model(input)
print(output)
m = nn.Sigmoid()
output = m(output)
print(output)
print(output.argmax(1))
print(test_dataset.classes)