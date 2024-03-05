import torchvision
from torch.utils.tensorboard import SummaryWriter
dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
trans_set = torchvision.datasets.CIFAR10(root="./datast1",train=True,transform=dataset_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root="./datast1",train=False,transform=dataset_transform,download=True )

img ,target = test_set[1]
print(test_set.classes)
print(img)
print(target)
print(test_set.classes[target])
writer = SummaryWriter("data")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set",img,i)
writer.close()

