import torch
import torch.nn as nn
import numpy as np

# 示例文本数据
texts = ['I love coding', 'Coding is fun', 'Python is my favorite programming language']

# 构建词汇表
word_to_idx = {}
for text in texts:
    for word in text.split():
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

vocab_size = len(word_to_idx)

#  看文本是否在词汇表中
def text_to_bow(text):
    bow = np.zeros(vocab_size)
    for word in text.split():
        if word in word_to_idx:
            bow[word_to_idx[word]] += 1
    return bow
 
# 示例输入数据
input_data = [text_to_bow(text) for text in  ['I love coding fun love']]
input_data = torch.tensor(input_data, dtype=torch.float32)

# 输出词袋向量
print(input_data)
