import torch
from torch import nn, optim
import torch.nn.functional as F

# 多层感知机模型

# 1：线性层 full-connected layer
# 实现一个线性回归模型 y = wx + b 
liner = nn.Linear(32, 1)
input = torch.rand(3, 32)
output = liner(input)
print(output.shape, liner.weight.shape, liner.bias.shape)
print(output)

# 2：激活函数-Sigmoid函数
 
x = torch.tensor([1.0, 2.0, 3.0])
# 应用Sigmoid函数
output = torch.sigmoid(x) #输出限定在[0,1]之间
print("Sigmoid：",output)

# 3：激活函数-softmax函数

x = torch.tensor([1.0, 2.0, 3.0])
output = F.softmax(x, dim=0)
print("softmax:",output)
print("sum:",output.sum()) # softmax函数的输出和为1

# 4：激活函数-Relu函数

x = torch.tensor([-1.0, 2.0, -3.0, 4.0])
output = F.relu(x)
print("Relu：",output)  # Relu函数的输出大于0，小于0的部分为0 

# 5：一个使用PyTorch实现逻辑回归的线性层的示例代码

import torch
import torch.nn as nn

# 创建逻辑回归模型类
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # 线性层

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))  # 应用sigmoid激活函数
        return out

# 定义输入数据的维度
input_dim = 10

# 创建逻辑回归模型实例
model = LogisticRegression(input_dim)

# 打印模型的结构
print(model)

# 创建输入张量
x = torch.randn(1, input_dim)

# 前向传播计算输出
output = model(x)

# 打印输出结果
print(output)
