from transformers import AutoTokenizer, BertModel
import torch

# 1. load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 2. tokenize sentence
# text = "Hello, my dog is cute"
text = "This is apple"
# text = ''
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
# 3. input tokens into BERT 
with torch.no_grad():
    outputs = model(**inputs)

# 4. get sentence embedding
sentence_embedding = outputs.last_hidden_state[0, 0].numpy()     # choose the first token in the first sentence, i.e., [CLS] token
# print(outputs.last_hidden_state.shape)     # [batch_size, sequence_length, hidden_size] hidden_size=768 in Bert Base
print(sentence_embedding.shape)
