# Defined in Section 4.6.6

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from vocab import Vocab
from utils import load_sentence_polarity
import sys 

class CnnDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

def collate_fn(examples):
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    # 对batch内的样本进行padding，使其具有相同长度
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, targets

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, filter_size, num_filter, num_class):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
 

        # filter_size 表示卷积核的大小（宽度）
        # num_filter 表示卷积核的数量
        # embedding_dim 表示词向量的维度
        self.conv1d = nn.Conv1d(embedding_dim, num_filter, filter_size, padding=1)
        self.activate = F.relu
        self.linear = nn.Linear(num_filter, num_class)
    def forward(self, inputs):
        embedding = self.embedding(inputs)
        # embedding.permute 将张量的维度换位

        '''
        默认情况下，embedding的形状是[batch_size, sequence_length, embedding_dim] 
        embedding.permute(0, 2, 1) 将其变为[batch_size, embedding_dim, sequence_length]
        因为 nn.Conv1d要求输入的张量形状是[batch_size, in_channels, sequence_length]，in_channels对应于输入的特征维度

        self.conv1d是一个nn.Conv1d层，它对输入进行一维卷积操作，卷积操作会在输入序列的维度上滑动一个固定大小的窗口（卷积核），并对窗口内的值进行加权和操作，生成卷积特征
        这里使用self.activate函数（通常是ReLU函数）对卷积特征进行非线性激活。

        F.max_pool1d(convolution, kernel_size=convolution.shape[2])对卷积特征进行一维最大池化操作
        。最大池化操作会从卷积特征中提取每个通道的最大值，用于汇总该通道的特征信息。kernel_size参数指定池化窗口的大小，
        这里使用convolution.shape[2]来动态设置池化窗口的大小，以确保窗口大小适应输入卷积特征的长度。
        '''
 
        convolution = self.activate(self.conv1d(embedding.permute(0, 2, 1)))
        pooling = F.max_pool1d(convolution, kernel_size=convolution.shape[2])
        outputs = self.linear(pooling.squeeze(dim=2))
        log_probs = F.log_softmax(outputs, dim=1)
        return log_probs

#tqdm是一个Pyth模块，能以进度条的方式显示迭代的进度
from tqdm.auto import tqdm

#超参数设置
embedding_dim = 128
hidden_dim = 256
num_class = 2
batch_size = 32
num_epoch = 5
filter_size = 3
num_filter = 100

#加载数据
train_data, test_data, vocab = load_sentence_polarity()
train_dataset = CnnDataset(train_data)
test_dataset = CnnDataset(test_data)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

#加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN(len(vocab), embedding_dim, filter_size, num_filter, num_class)
model.to(device) #将模型加载到CPU或GPU设备

#训练过程
nll_loss = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) #使用Adam优化器

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
        inputs, targets = [x.to(device) for x in batch]
        log_probs = model(inputs)
        loss = nll_loss(log_probs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss: {total_loss:.2f}")

#测试过程
acc = 0
for batch in tqdm(test_data_loader, desc=f"Testing"):
    inputs, targets = [x.to(device) for x in batch]
    with torch.no_grad():
        output = model(inputs)
        acc += (output.argmax(dim=1) == targets).sum().item()

#输出在测试集上的准确率
print(f"Acc: {acc / len(test_data_loader):.2f}")
