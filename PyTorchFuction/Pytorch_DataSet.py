from torch.utils.data import Dataset
import  os
from PIL import Image

class MyDataset(Dataset):

    def __init__(self,  root_dir,label_dir):
        """

        :param root_dir:图像数据的文件夹位置
        :param label_dir: 图像数据文件夹中所对应图像的标签位置
        """
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = root_dir
        self.img_path = os.listdir(self.path)
        self.img_label = os.listdir(self.label_dir)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_label = self.img_label[idx]
        img_name_path = os.path.join(self.root_dir, img_name)
        img_label_path = os.path.join(self.label_dir,img_label)
        img = Image.open(img_name_path)
        label = open(img_label_path)
        return img, label

    def __len__(self):
        return len(self.img_path)

ants_root_dir ="train/ants_image"
bees_root_dir ="train/bees_image"
ants_label_dir = "train/ants_label"
bees_label_dir = "train/bees_label"
ants_dataset = MyDataset(ants_root_dir, ants_label_dir)
bees_dataset = MyDataset(bees_root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset
print(ants_dataset)