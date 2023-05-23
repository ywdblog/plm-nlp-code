
import torch
import torch.nn as nn

# 第三章：文本表示法2

# 1：词的分布式-词嵌入（Word Embedding）
# - 语义信息
# - 上下文信息
# - 向量运算

# 定义词嵌入模型
class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input):
        embedded = self.embedding(input)
        return embedded

# 示例输入数据
input_data = torch.tensor([1, 2, 3, 4, 5])  # 假设共有5个词

# 定义模型和参数
vocab_size = 6  # 词汇表大小，包括一个未登录词（out-of-vocabulary）
embedding_dim = 100  # 词嵌入的维度
model = WordEmbedding(vocab_size, embedding_dim)

# 输出词嵌入结果
embedded = model(input_data)
print(embedded)
