from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

#使用方式1 转换成tensor数据格式
img_path=r'C:\Users\86187\Desktop\pytorch_test\hymenoptera_data\train\ants\6240329_72c01e663e.jpg'
img=Image.open(img_path)
tran_tensor=transforms.ToTensor() #创建ToTensor 类 Transforms中的Totensor工具
img_tensor=tran_tensor(img) #将PIL.jpg格式转换为tensor格式 包装了神经网络中的一些参数 肯定要转换成tensor型进行训练
print(type(img_tensor))#此时验证是tensor的工具

writer=SummaryWriter('../logs')
writer.add_image('tensor_img',img_tensor)

#使用方式2 归一化 图片,矩阵           mean          std
trans_norm=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]) #分别填入均值和标准差 有三通道要填三层
img_norm =trans_norm(img_tensor)

#使用方式3 resize等比缩放图片
#PIL - RESIZE- TENSOR
print(img_tensor.size())
trans_resize=transforms.Resize((512,512)) #将图片大小重新变成512*512 还是类
img_resize=trans_resize(img) #调用方法
img_resize=tran_tensor(img_resize) #转换成tensor数据类型
writer.add_image('test',img_resize)

print(img_resize)

#使用方式4 随机裁剪 tranforms.RamdomCrop
trans_random=transforms.RandomCrop((20,30)) #裁剪图片大小
for i in range(10):
    img_crop=trans_random(img_resize) #需要用的的tensor类型的
    writer.add_image('randomcrop',img_crop,i) #i是step

#使用方式5 compose 载入图片
trans_compose=transforms.Compose([tran_tensor,trans_resize])
img_compose=trans_compose(img)
