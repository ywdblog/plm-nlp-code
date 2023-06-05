
from torch.nn import RNN
import torch

# 第四章：循环神经网络 RNN
# 参考 lstm_sent_polarity.py

# 原始循环神经网络：
# 当使用循环神经网络处理一个序列输入时，需要将循环神经网络按输入时刻展开，然后将序列中的每个输入依次对应到网络不同时刻的输入上，并将当前时刻网络隐含层的输出也作为下一时刻的输入
# tanh函数是一个S型激活函数，它的取值范围是(-1,1) ，是输入序列的当前时刻，其隐含层ht不但与当前的输入xt有关，而且与上一时刻的隐含层ht−1有关
# 每个时刻的输入经过层层递归，对最终的输出产生一定的影响，每个时刻的隐含层ht承载了1 ∼ t时刻的全部输入信息，因此循环神经网络中的隐含层也被称作记忆单元

# 长短时记忆网络：
# 避免多个隐含层导致的损失，LSTM跨过了中间的t−k层，从而减小了网络的层数，使得网络更容易被优化

# 双向LSTM
# 传统的循环神经网络并不能利用某一时刻后面的信息
# 双向LSTM可以同时利用当前时刻之前和之后的信息，从而更好地捕捉序列中的依赖关系

# 1：原始循环神经网络

# 1）：构造
# 构造一个输入序列长度为3，输入维度为4，隐含层维度为5的循环神经网络
# hidden_size: 隐含层的维度 input_size: 输入的维度（单词个数）
run = RNN(input_size=4, hidden_size=5,  batch_first=True )

# 2）：输入
# 生成输入数据 batch ,seq_len, input_size
inputs = torch.rand(2, 3, 4) #序列长度是3，输入维度是4（单词个数）

# 3）：输出
outputs ,hn = run(inputs)

# 第一个输出表示隐含层序列，shape是(2, 3, 5)，表示两个样本，每个样本有3个时刻，每个时刻的隐含层维度是5
print("outputs:",outputs, outputs.shape)  

# 第二个输出表示最后一个时刻的隐含层，shape是(1, 2, 5)，表示两个样本，每个样本有1个时刻，每个时刻的隐含层维度是5
print("hn:",hn, hn.shape)

# 2：长短时记忆网络

from torch.nn import LSTM
lstm = LSTM(input_size=4, hidden_size=5, batch_first=True)
inputs = torch.rand(2, 3, 4)
outputs, (hn, cn) = lstm(inputs) #cn是最后一个时刻的隐含层状态

