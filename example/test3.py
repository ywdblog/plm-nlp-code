
pip install gensim

from gensim.models import Word2Vec

# 示例数据集
sentences = [["I", "love", "machine", "learning"],
             ["Word", "embeddings", "are", "awesome"],
             ["NLP", "is", "fun"]]

# 训练Word2Vec模型
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 保存模型
#model.save("word2vec_model.bin")

# 加载已训练的模型
model = Word2Vec.load("word2vec_model.bin")

# 获取单词"machine"的词向量
word_vector = model.wv['machine']
print(f"Word 'machine' Vector: {word_vector}")
