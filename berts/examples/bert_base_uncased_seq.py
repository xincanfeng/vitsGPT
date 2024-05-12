from transformers import AutoTokenizer, BertModel
import torch

# 1. load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 2. tokenize sentence
text = "Hello, my dog is cute"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
# 3. input tokens into BERT
with torch.no_grad():
    outputs = model(**inputs)

# 4. get sentence embedding
sentence_embedding = outputs.last_hidden_state[
    0
].numpy()  # choose the tokens for the first sentnece, i.e., seq token（including cls and sep）
# print(outputs.last_hidden_state)     # [batch_size, sequence_length, hidden_size] hidden_size=768 in Bert Base
print(sentence_embedding.shape)
