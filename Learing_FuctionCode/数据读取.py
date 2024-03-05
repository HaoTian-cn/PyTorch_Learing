from torch.utils.data import Dataset
from PIL import  Image
import os
class mydata(Dataset):

    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir #一般是源地址
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir) #将两个地址相加
        self.img_path=os.listdir(self.path) #获取图片名称列表

    def __getitem__(self, item): #item 表示编号  包含input label 该函数指的是 返回实例对象的属性
        img_name=self.img_path[item] #图片名称
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)  #每一个图片的位置
        img=Image.open(img_item_path)
        label=self.label_dir #实例名称 "是什么"
        return img,label
    def __len__(self): #长度返回
        return len(self.img_path)
root_dir=r'C:\Users\86187\Desktop\pytorch_test\hymenoptera_data\train'
label_dir='ants' #文件夹名称
ants_dataset=mydata(root_dir,label_dir) #获取蚂蚁实例
bees_dataset=mydata(root_dir,'bees') #获取蜜蜂数据
all_data=ants_dataset+bees_dataset
print(len(all_data))