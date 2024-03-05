import torch
import torchvision
vgg16=torchvision.models.vgg16()

#保存方式1
#保存了模型的结构和模型的参数
torch.save(vgg16,'../vgg_model.pth')
#调用方式
import torch
torch.load('../vgg16_model.pth')

#保存方式2
#只保存模型的参数 官方推荐的保存方式 ，模型的空间比较小 **
torch.save(vgg16.state_dict(),'../vgg16_model2')
#调用方式
import torch
import torchvision
vgg16_2=torchvision.models.vgg16() #vgg16原模型
vgg16_2.load_state_dict(torch.load('../vgg16_model2.pth')) #调用字典
# from **(py文件名) import * 将**中的所有东西引入过来