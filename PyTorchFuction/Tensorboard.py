from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
writer = SummaryWriter("logs")
img_path = "dataset/catpic.jpg"
img = Image.open(img_path)
print(type(img))
img_array = np.array(img)
writer.add_image("test",img_array,1,dataformats='HWC')
for i in range(100):
    writer.add_scalar("y=3x",i*3,i)

writer.close()
