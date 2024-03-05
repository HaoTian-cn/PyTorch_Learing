from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
writer=SummaryWriter(r'C:\Users\86187\Desktop\pytorch_test\logs') #创建实例存储到logs文件夹底下 打开上级目录
#图片的图
img_path=r'C:\Users\86187\Desktop\pytorch_test\hymenoptera_data\train\ants\6240338_93729615ec.jpg' #这里用的绝对路径
img=Image.open(img_path)
img_numpy=np.array(img) #将img转为numpy类型
print(img_numpy.shape)
writer.add_image("test",img_numpy,2,dataformats='HWC')  # 添加image 1代表step 就是滑块中的第几个位置 test是名称
#writer.add_image('',tensor_type)可以直接输出
#函数的图
for i in range(100):
    writer.add_scalar('y=2x',2*i,i)  #添加scalar “y=2*x”是名称 然后是y值然后是x值
writer.close()
'''
运行完了后去虚拟终端 运行 tensorboard --logdir=logs
'''