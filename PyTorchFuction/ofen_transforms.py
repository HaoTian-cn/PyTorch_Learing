from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


# ToTensor 的使用
writer = SummaryWriter("logs")
Img = Image.open("dataset/catpic.jpg")
Img_totensor = transforms.ToTensor()
img_1 = Img_totensor(Img)
img = writer.add_image("test",img_1)
#Normalize 的使用
trans_img = transforms.Normalize([1,3,5],[0.5,0.5,0.5])
img_nom = trans_img(img_1)
img_nom1 = writer.add_image("nomalise",img_nom,2)
#Resize 的使用
trans_resize = transforms.Resize((512,512))
# img_PIL -> risize -> img_resize PIL
img_2 = trans_resize(Img)
print(img_2)
#img_PIL -> img_totensor ->img_resize_tensor
img_3 = Img_totensor(img_2)
img_3 = trans_resize(img_3)
print(type(img_3))
#Compose -> resize -> method 2
trans_resize_2 = transforms.Resize((512,512))
trans_compose = transforms.Compose([trans_resize_2,Img_totensor])#将PIL 未修剪的图片直接转换为tensor类型且修剪过后的图片
img_trans_2 = trans_compose(Img)
writer.add_image("compose",img_trans_2,1)
writer.close()