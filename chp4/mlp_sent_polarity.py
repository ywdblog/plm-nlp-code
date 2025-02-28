# Defined in Section 4.6.4 and 4.6.5 mlp_sent_polarity.py

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from vocab import Vocab
from utils import load_sentence_polarity
import sys

class BowDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

# inputs 表示输入的词的索引，offsets 表示每个句子的起始位置，targets 表示情感标签
def collate_fn(examples):
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    # 将不定长的序列拼接起来，然后使用一个偏移向量（Offsets）记录每个序列的起始位置
    offsets = [0] + [i.shape[0] for i in inputs]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    inputs = torch.cat(inputs)
    return inputs, offsets, targets

class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(MLP, self).__init__()
        # EmbeddingBag
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.activate = F.relu
        self.linear2 = nn.Linear(hidden_dim, num_class)
    def forward(self, inputs, offsets):
        embedding = self.embedding(inputs, offsets)
        hidden = self.activate(self.linear1(embedding))
        outputs = self.linear2(hidden)
        log_probs = F.log_softmax(outputs, dim=1)
        return log_probs

# tqdm是一个Python模块，能以进度条的方式显示迭代的进度
from tqdm.auto import tqdm

# 超参数设置
embedding_dim = 128
hidden_dim = 256
num_class = 2
batch_size = 32
num_epoch = 5

# 加载数据
train_data, test_data, vocab = load_sentence_polarity()
# dataset 用于存储数据 
train_dataset = BowDataset(train_data)
test_dataset = BowDataset(test_data)

# collate_fn参数指向一个函数，用于对一个批次的样本进行整理，如将其转换为张量等
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(len(vocab), embedding_dim, hidden_dim, num_class)
model.to(device) # 将模型加载到CPU或GPU设备

#训练过程
nll_loss = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # 使用Adam优化器

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
        inputs, offsets, targets = [x.to(device) for x in batch]
        # model 第一个参数是输入的词的索引，第二个参数是每个句子的起始位置
        log_probs = model(inputs, offsets)
        # 计算损失
        loss = nll_loss(log_probs, targets)
        # optimizer.zero_grad() 将模型参数的梯度置零，然后进行反向传播，最后更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # loss.item() 返回一个标量，保存了损失的值
        total_loss += loss.item()
    print(f"Loss: {total_loss:.2f}")

# 测试过程
acc = 0
for batch in tqdm(test_data_loader, desc=f"Testing"):
    inputs, offsets, targets = [x.to(device) for x in batch]
    # torch.no_grad() 告诉PyTorch不要计算梯度，这样可以节省内存
    with torch.no_grad():
        output = model(inputs, offsets)
        # dim=1 表示按行计算，即计算每一行的最大值
        acc += (output.argmax(dim=1) == targets).sum().item()

# 输出在测试集上的准确率
print(f"Acc: {acc / len(test_data_loader):.2f}")
