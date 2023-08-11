from transformers import BertTokenizer, BertModel

# 加载预训练的BERT模型和对应的tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 输入文本
text = "Hugging Face is a great platform for NLP models."

# 对文本进行tokenize和编码
inputs = tokenizer(text, return_tensors="pt")

# 使用BERT模型获取文本的词嵌入（即输出的隐藏状态）
outputs = model(**inputs)

# 获取文本的词嵌入向量
word_embeddings = outputs.last_hidden_state

print(f"Input Text: {text}")
print("Word Embeddings Shape:", word_embeddings.shape)
