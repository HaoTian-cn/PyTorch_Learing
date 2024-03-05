import torchvision
from torch.utils.data import DataLoader
#准备测试的数据集
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10(root="./datast1",train=False,transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_set,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
#测试第一张数据集的图片及target
img ,target = test_set[0]
print(img.shape)
print(target)
writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images("epoch:{}".format(epoch),imgs,step)
        step = step + 1
writer.close()