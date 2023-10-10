from transformers import AutoModel, AutoTokenizer
from text2phonemesequence import Text2PhonemeSequence
import torch

# Load XPhoneBERT model and its tokenizer
xphonebert = AutoModel.from_pretrained("vinai/xphonebert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/xphonebert-base")

# Load Text2PhonemeSequence
text2phone_model = Text2PhonemeSequence(language='eng-us', is_cuda=True)
# text2phone_model = Text2PhonemeSequence(language='jpn', is_cuda=True)

# Input sequence that is already WORD-SEGMENTED (and text-normalized if applicable)
sentence = "Hello, my dog is cute"
# sentence = "これ は 、 テスト テキスト です ."

input_phonemes = text2phone_model.infer_sentence(sentence)
input_ids = tokenizer(input_phonemes, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    features = xphonebert(**input_ids)
    
sentence_embedding = features.last_hidden_state[0].numpy()
print(sentence_embedding.shape)

