#dataloader 的使用方法
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision
import shutil #删除文件夹的模块
shutil.rmtree('../logs')
writer=SummaryWriter('../logs')
test_data=torchvision.datasets.CIFAR10('../dataset',train=False,transform=torchvision.transforms.ToTensor())
#dataloader处理数据 ，打包四个，随机打乱
test_loder=DataLoader(dataset=test_data,batch_size=64,shuffle=True,drop_last=False)
#batch_size 每次取四个数据集进行打包 shuffle 随机打乱数据 drop_last=Ture如果打包不够的话就会自动舍去数据
img,target=test_data[0]
print(img.shape)
print(target)
step=0
for data in test_loder:
    img,target=data
    # print(img.shape)
    # print(target)
    writer.add_images('test_data',img,step)#这里的代码是add_images
    step+=1