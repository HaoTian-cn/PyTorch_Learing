import os
import pandas as pd
import numpy as np
#复现代码仅仅采用了35Hz12kN西安交大的数据集
#第一列为水平振动信号，第二列为垂直振动信号
rootdir = r'C:\Users\86187\Desktop\数据集\XJTU-SY\data\XJTU-SY_Bearing_Datasets\35Hz12kN'
class Dataset():
    def __init__(self,rootdir):
        self.rootdir=rootdir
        self.listdir1=os.listdir(rootdir)
    def __getitem__(self, item):
        data=np.array([np.nan,np.nan])
        list2=os.listdir(os.path.join(self.rootdir,self.listdir1[item]))
        list2=sorted(list2,key= lambda x:int(x.split('.')[0]))

        for path_last in list2:
            df=pd.read_csv(os.path.join(self.rootdir,self.listdir1[item],path_last),encoding='gbk')
            df=np.array(df.values)
            data=np.vstack((data,df))
        self.num=len(data)

        return data
    def __len__(self):
        return self.num
if __name__=='__main__':
    rootdir=r'C:\Users\86187\Desktop\复现\35Hz12kN'
    dataset=Dataset(rootdir)
    data=dataset[0][~np.isnan(dataset[0]).any(axis=1)]
    # for data_item in dataset:
    #     data=np.vstack((data,data_item))
    # data=data[~np.isnan(data).any(axis=1)] #删除nan行
    print('数据读取完成,数据长度为:',data.shape[0])


