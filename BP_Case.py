import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 生成随机数据
np.random.seed(42)
inputs = np.random.rand(1000, 401)  # 1000个样本，每个样本有401个特征
targets = np.random.rand(1000, 1)  # 对应的目标值

# 转换为 PyTorch 张量
inputs = torch.tensor(inputs, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(401, 128)  # 输入层到隐藏层
        self.fc2 = nn.Linear(128, 1)   # 隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用ReLU激活函数
        x = self.fc2(x)
        return x

# 创建模型实例
model = SimpleNet()

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器

# 训练模型
epochs = 100  # 训练轮数
for epoch in range(epochs):


    outputs = model(inputs)  # 前向传播
    loss = criterion(outputs, targets)  # 计算损失
    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 反向传播

    optimizer.step()  # 更新权重

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
