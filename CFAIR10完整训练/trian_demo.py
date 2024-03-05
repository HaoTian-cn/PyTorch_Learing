import os
import time
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from module import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device='cuda'
#  准备数据集
train_dataset = torchvision.datasets.CIFAR10("train_data", train=True,transform=torchvision.transforms.ToTensor(), download=True,)
test_dataset = torchvision.datasets.CIFAR10("test_data", train=False, transform=torchvision.transforms.ToTensor(), download=True,)

#  数据的length（长度)
train_dataset_length  = len(train_dataset)
test_dataset_length = len(test_dataset)
print(test_dataset_length)
print(train_dataset_length)

# 利用DataLoader 加载所需的数据
train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, drop_last=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, drop_last=False)

#  创建网络模型
model = mymodule().cuda()
#  设置损失函数
loss_fn = torch.nn.CrossEntropyLoss().cuda()
#  设置优化器
#  设置学习速率
learning_rate = 1e-3
optimer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#  设置训练网络的一些参数
#  记录训练的次数
total_train_step = 0
#  记录测试的次数
total_test_step = 0
#  记录训练的运行轮数
total_train_epoch = 20
#  添加tensorboard
writer = SummaryWriter("train")
#  训练步骤开始

for i in range(total_train_epoch):
    print("------第{}轮训练开始------".format(i+1))

    model.train()
    start_time = time.time()
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        output = model(imgs)
        #   计算实际与结果误差函数
        loss = loss_fn(output,targets)
        #   优化器优化模型
        optimer.zero_grad()
        #   反向传播
        loss.backward()
        optimer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("已经训练次数：{},其total_train_loss：{}".format(total_train_step,loss.item()))
            #writer.add_scalar("loss_value", loss.item(), global_step=total_train_step)
            pass
    end_time = time.time()
    print("运行时间：{}".format(end_time - start_time))
    #  测试步骤开始
    model.eval()
    #  设置判断正确的个数
    accuracy = 0
    #   设置正确预测个数
    total_test_accuracy = 0
    #  设置损失函数计算的差值
    total_test_loss  =  0
    #  消除梯度对模型的预测影响
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            output = model(imgs)
            loss = loss_fn(output,targets)
            total_test_loss = total_test_loss + loss.item()
            #  计算预测图片的正确率
            accuracy = (output.argmax(1)==targets).sum()
            total_test_accuracy = total_test_accuracy + accuracy
    print("整体测试集上的正确率:{}".format(total_test_accuracy / test_dataset_length))
    print("整体测试集上的Loss:{}".format(total_test_loss))
    #writer.add_scalar("test_loss_value",total_test_loss,global_step=total_test_step)
    #writer.add_scalar("test_accuracy_value", accuracy / test_dataset_length, global_step=total_test_step)
    total_test_step = total_test_step + 1
    total_train_loss = 0
torch.save(model,"module_{}.pth".format(total_train_epoch-1))
print("模型已保存")

writer.close()

