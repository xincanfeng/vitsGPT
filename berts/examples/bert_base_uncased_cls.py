from transformers import AutoTokenizer, BertModel
import torch

# 1. 选择并加载预训练的 BERT 模型和分词器
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 2. 将句子转换为 tokens
# text = "Hello, my dog is cute"
text = "This is apple"
# text = ''
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
# 3. 将 tokens 输入到 BERT 模型
with torch.no_grad():
    outputs = model(**inputs)

# 4. 获取句子的嵌入表示
sentence_embedding = outputs.last_hidden_state[0, 0].numpy()     # 选择第一个句子的第一个token，即[CLS]token
# print(outputs.last_hidden_state.shape)     # [batch_size, sequence_length, hidden_size] hidden_size=768 in Bert Base
print(sentence_embedding.shape)
