import torch 

# 第四章：CNN 卷积神经网络
# 参考 cnn_sent_polarity.py

from torch.nn import Conv1d 
from torch.nn import MaxPool1d
import torch.nn.functional as F
import sys 

# 卷积核的输出长度计算公式=  单词的个数 - 卷积核的宽度 + 1

# kernel_size 为卷积核的宽度
# out_channels 为卷积核的个数（输出的通道数）
# in_channels 为输入的通道数，输入层对应词向量的维度
conv1 = Conv1d(in_channels=5, out_channels=2, kernel_size=4)
conv2 = Conv1d(in_channels=5, out_channels=2, kernel_size=3)

# 定义了两个Conv1d都原因是：卷积核的宽度不同，输出的长度也不同

# 1：构造

# 生成输入数据 batch ,in_channels, input_length
inputs = torch.randn(2, 5, 6) # batch_size=2, 输入通道数=5（输入维度）, 输入长度=6（单词个数）
# 我，喜欢，自然，语言，处理，。  6个单词，每个单词用5维的词向量表示
print(inputs)

# 计算输出 batch ,out_channels, output_length
out1 = conv1(inputs)
out2 = conv2(inputs)
print("out1:",out1, out1.shape) # 2, 2, 3 
print("out2:",out2, out2.shape) # 2, 2, 4
# 输出长度 = 输入长度 - kernel_size + 1

# 2：池化层

pool1 = F.max_pool1d(out1, kernel_size=out1.shape[2]) # 池化层核的大小的设置为输出的长度   
pool2 = F.max_pool1d(out2, kernel_size=out2.shape[2]) # 池化层核的大小的设置为输出的长度
print("pool1:",pool1, pool1.shape) # 2, 2, 1
print("pool2:",pool2, pool2.shape) # 2, 2, 1
# max_pool1d 的维度为  batch ,out_channels, 1
 
# 3: 拼接

# 将两行一列的矩阵转换为向量，然后拼接
out1_pool_squeeze1 = torch.squeeze(pool1,dim=2) # 去掉维度为1的维度
out1_pool_squeeze2 = torch.squeeze(pool2,dim=2)
output_pool = torch.cat((out1_pool_squeeze1, out1_pool_squeeze2), dim=1) # 拼接
print("output:",output_pool, output_pool.shape)  # 2, 4

# 4：全连接层

from   torch.nn import Linear
fc = Linear(in_features=4, out_features=2) # 输入维度为4，输出维度为2
output = fc(output_pool)
print("output:",output, output.shape) # 2, 2