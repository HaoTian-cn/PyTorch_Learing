from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
img_path = "dataset/catpic.jpg"
img = Image.open(img_path)
img_totensor = transforms.ToTensor()
imgtensor = img_totensor(img)#将image类型 转换为 tensor类型
print(imgtensor)
writer = SummaryWriter("logs")
img_tensor = writer.add_image("img_tensor",imgtensor,1,dataformats="CHW")
writer.close()