from transformers import AutoModel, AutoTokenizer
from text2phonemesequence import Text2PhonemeSequence
import torch
import sys

# 使用 sys.argv 来获取命令行参数
dataset = sys.argv[1]

# Ensure that GPU is available and set the device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise ValueError("No GPU found!")

# Load XPhoneBERT model and its tokenizer
xphonebert = AutoModel.from_pretrained("vinai/xphonebert-base").to(device)
tokenizer = AutoTokenizer.from_pretrained("vinai/xphonebert-base")

# Load Text2PhonemeSequence
text2phone_model = Text2PhonemeSequence(language='eng-us', is_cuda=True)

def get_embedding(sentence):
    input_phonemes = text2phone_model.infer_sentence(sentence)
    input_ids = tokenizer(input_phonemes, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        features = xphonebert(**input_ids)
    # Get the sentence sequential embedding and move it to CPU in float16 format
    sentence_embedding = features.last_hidden_state[0].to('cpu').half()
    return sentence_embedding

# Read from text file and compute embeddings
# dataset = "ljs"
# dataset = "librif"
# dataset = "emovdb"
filelist_dir = "/data/vitsGPT/vits/filelists/"
file_path = f"{filelist_dir}{dataset}_audio_text_all_filelist.txt"  

embeddings_dict = {}
with open(file_path, 'r') as file:
    for line in file:
        audiopath, sentence = line.strip().split('|')
        embeddings_dict[audiopath] = get_embedding(sentence)

# Save embeddings to a .pt file
torch.save(embeddings_dict, f"{filelist_dir}{dataset}_audio_bert_phone_768.pt")  # Save sequential embeddings to a .pt file
