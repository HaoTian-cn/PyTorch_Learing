#视觉数据集
import torchvision
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter('../logs')
train_set=torchvision.datasets.CIFAR10(root='../dataset',download=True,train=True) #载入CIFAR10数据集
test_set=torchvision.datasets.CIFAR10(root='../dataset',train=False,download=True) #载入测试集
#每个集合分别是 用 图片 序号 来储存 所以img返回图片 target返回data中classes类的名称序号
# dataset_transform=torchvision.transforms.Compose([torchvision.transforms.Resize(100),torchvision.transforms.ToTensor()])
dataset_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
i=0
for img,target in train_set:
    img_compose=dataset_transform(img)
    writer.add_image('train',img_compose,i)
    i+=1
    print('第{}张图片加载完成'.format(i))
    # if i >50:
    #     break
writer.close()

