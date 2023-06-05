import torch
from torch import nn, optim
from torch.nn import functional as F

# 第四章：词向量&词袋模型&小批次梯度下降

# 1：词向量层
# 词表大小为8，嵌入维度为3
embedding = nn.Embedding(8, 3)
input = torch.tensor([[1, 2, 1, 0], [5, 6, 7, 4]], dtype=torch.long)
output = embedding(input)

# 2：融入词向量层的多层感知器

# 一个序列中通常含有多个词向量，那么如何将它们表示为一个多层感知器的输入向量呢？
# 一种方法是将n个向量拼接成一个大小为n×d的向量，其中d表示每个词向量的大小
# 但在一个序列前面添加一个标记，则序列中的每个标记位置都变了，所以一般使用词袋模型

class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(MLP, self).__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 线性变换：词嵌入层->隐含层
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        # 使用ReLU激活函数
        self.activate = F.relu
        # 线性变换：激活层->输出层
        self.linear2 = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        # 将序列中多个embedding进行聚合（此处是求平均值）
        embedding = embeddings.mean(dim=1)
        hidden = self.activate(self.linear1(embedding))
        outputs = self.linear2(hidden)
        # 获得每个序列属于某一类别概率的对数值
        probs = F.log_softmax(outputs, dim=1)
        return probs

mlp = MLP(vocab_size=8, embedding_dim=3, hidden_dim=5, num_class=2)
# 输入为两个长度为4的整数序列
inputs = torch.tensor([[0, 1, 2, 1], [4, 6, 6, 7]], dtype=torch.long)
outputs = mlp(inputs)
#print(outputs)

# 3: 词袋模型
# 在实际的自然语言处理任务中，一个批次里输入的文本长度往往是不固定的，因此无法像上面的代码一样简单地用一个张量存储词向量并求平均值
# PyTorch提供了一种更灵活的解决方案，即EmbeddingBag层。
# 在调用Embedding-Bag层时，首先需要将不定长的序列拼接起来，然后使用一个偏移向量（Offsets）记录每个序列的起始位置

input1 = torch.tensor([0,1,2,1], dtype=torch.long)
input2 = torch.tensor([2,1,3,7,5], dtype=torch.long)
input3 = torch.tensor([6,4,2], dtype=torch.long)
input4 = torch.tensor([1,3,4,3,5,7], dtype=torch.long)
inputs = [input1, input2, input3, input4]
offsets = [0] + [i.shape[0] for i in inputs]  
print(offsets)
# 下面的代码将输入的词向量拼接成一个张量
offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
inputs = torch.cat(inputs)
embeddingbag =  nn.EmbeddingBag(num_embeddings=8, embedding_dim=3 )
# embeddingbag的输入为两个张量，第一个张量为词向量，第二个张量为偏移向量
embeddings = embeddingbag(inputs, offsets)
print("词袋模型:",embeddings)
 
# 4：小批次梯度下降
# import torch
from torch import nn, optim
from torch.nn import functional as F

# 小批次梯度下降

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activate = F.relu
        self.linear2 = nn.Linear(hidden_dim, num_class)

    #forward函数定义了模型的前向传播过程，输入为x，输出为log_probs
    def forward(self, inputs):
        hidden = self.linear1(inputs)
        activation = self.activate(hidden)
        outputs = self.linear2(activation)
        # 获得每个输入属于某一类别的概率（Softmax），然后再取对数
        # 取对数的目的是避免计算Softmax时可能产生的数值溢出问题
        log_probs = F.log_softmax(outputs, dim=1)
        return log_probs

# 异或问题的4个输入
x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
# 每个输入对应的输出类别
y_train = torch.tensor([0, 1, 1, 0])

# 创建多层感知器模型，输入层大小为2，隐含层大小为5，输出层大小为2（即有两个类别）
model = MLP(input_dim=2, hidden_dim=5, num_class=2)

criterion = nn.NLLLoss() # 当使用log_softmax输出时，需要调用负对数似然损失（Negative Log Likelihood，NLL）
optimizer = optim.SGD(model.parameters(), lr=0.05) # 使用梯度下降参数优化方法，学习率设置为0.05

for epoch in range(100):
    y_pred = model(x_train) # 调用模型，预测输出结果
    loss = criterion(y_pred, y_train) # 通过对比预测结果与正确的结果，计算损失
    optimizer.zero_grad() # 在调用反向传播算法之前，将优化器的梯度值置为零，否则每次循环的梯度将进行累加
    loss.backward() # 通过反向传播计算参数的梯度
    optimizer.step() # 在优化器中更新参数，不同优化器更新的方法不同，但是调用方式相同

print("Parameters:")
for name, param in model.named_parameters():
    print (name, param.data)

y_pred = model(x_train)
print("Predicted results:", y_pred.argmax(axis=1))
