import torch 

# 第三章：文本表示法1

# 1： 使用 torch 实现 one-hot 编码
import torch

# labels = torch.tensor([2, 0, 1, 3, 4])  # 分类变量的取值

# num_classes = labels.max() + 1  # 类别的总数
# one_hot = torch.eye(num_classes)[labels]  # 创建单位矩阵并选择对应的行

# print(one_hot)

# 2：对一个文本进行 one-hot 编码 
# one-hot 没有考虑序列的顺序，也没有考虑单词之间的关系，因此无法表达单词之间的关系，也无法表达单词的相似性
# 可以使用更高级的编码方式，如词袋模型、词嵌入等

# import string

# # 定义文本
# text = "Hello, world! This is a sample text."

# # 去除标点符号并转换为小写
# text = text.translate(str.maketrans('', '', string.punctuation)).lower()

# # 创建字符到索引的映射
# char_to_index = {char: i for i, char in enumerate(set(text))}

# # 字符总数和独特字符数
# num_chars = len(text)
# num_unique_chars = len(char_to_index)

# # 创建一个大小为 (num_chars, num_unique_chars) 的全零张量
# one_hot = torch.zeros(num_chars, num_unique_chars)

# # 将文本转换为 One-hot 编码
# for i, char in enumerate(text):
#     char_index = char_to_index[char]
#     one_hot[i, char_index] = 1

# print(one_hot)

# 3：词袋模型
# 词袋模型是一种简单的文本表示方法，它将文本表示为一个向量，向量中的每个元素表示一个单词在文本中出现的次数
# 词袋模型忽略了单词出现的顺序，因此无法表达单词之间的关系，也无法表达单词的相似性

# import string

# # 定义文本
# text = "Hello, world! This is a sample text. Hello, world again!"

# # 去除标点符号并转换为小写
# text = text.translate(str.maketrans('', '', string.punctuation)).lower()

# # 创建词汇表
# words = text.split()
# vocab = list(set(words))
# vocab_size = len(vocab)

# # 创建词汇表到索引的映射
# word_to_index = {word: i for i, word in enumerate(vocab)}
# print(word_to_index)

# # 创建一个大小为 (1, vocab_size) 的全零张量
# bag_of_words = torch.zeros(1, vocab_size)

# # 将文本转换为词袋模型表示
# for word in words:
#     word_index = word_to_index[word]
#     bag_of_words[0, word_index] += 1

# print(bag_of_words)

# 4：multi-hot 实现

# import torch
# import string
# import sys

# # 定义文本
# text = "Hello, world! This is a sample text. Hello, world again!"

# # 去除标点符号并转换为小写
# text = text.translate(str.maketrans('', '', string.punctuation)).lower()

# # 创建词汇表
# words = text.split()
# vocab = list(set(words))

# # 创建词汇表到索引的映射
# word_to_index = {word: i for i, word in enumerate(vocab)}
# print(word_to_index)

# # 创建一个大小为 (1, vocab_size) 的全零张量
# multi_hot = torch.zeros(1, len(vocab))
 
# test_text = "Hello, world again!"
# test_text = test_text.translate(str.maketrans('', '', string.punctuation)).lower()
# test_word = test_text.split()
# print(test_word)
# # 将文本转换为多热编码
# for word in test_word:
#     if word in word_to_index:
#         word_index = word_to_index[word]
#         multi_hot[0, word_index] = 1

# print(multi_hot)

# 5：使用 torch 实现 multi_hot

# import torch
# from torch.nn.functional import one_hot
# import string

# # 定义文本
# text = "Hello, world! This is a sample text. Hello, world again!"

# # 去除标点符号并转换为小写
# text = text.translate(str.maketrans('', '', string.punctuation)).lower()

# # 创建词汇表
# words = text.split()
# vocab = list(set(words))

# # 创建词汇表到索引的映射
# word_to_index = {word: i for i, word in enumerate(vocab)}

# # 将文本转换为多热编码
# encoded_text = [word_to_index[word] for word in words if word in word_to_index]
# multi_hot = one_hot(torch.tensor(encoded_text), num_classes=len(vocab)).sum(dim=0)

# print(multi_hot)

# 6：另外一个构建词袋模型的方法
# - 构建词汇表
# - 特征提取
# - 特征向量化

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

# 特征提取
def text_to_bow(text):
    bow = np.zeros(vocab_size)
    for word in text.split():
        if word in word_to_idx:
            bow[word_to_idx[word]] += 1
    return bow

# 示例输入数据
input_data = [text_to_bow(text) for text in texts]
input_data = torch.tensor(input_data, dtype=torch.float32)

# 输出词袋向量
print(input_data)
