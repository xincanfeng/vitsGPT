from transformers import AutoTokenizer, BertModel
import torch
import sys

# use sys.argv to get command parameters
dataset = sys.argv[1]

# Ensure that GPU is available and set the device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise ValueError("No GPU found!")

# 1. Load pretrained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name).to(device)  # Move model to GPU


def get_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(
        device
    )  # Move inputs to GPU
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the sentence embedding and move it to CPU in float16 format
    sentence_embedding = outputs.last_hidden_state[0, 0].to("cpu").half()
    return sentence_embedding


# Read from text file and compute embeddings
# dataset = "ljs"
# dataset = "librif"
# dataset = "emovdb"
filelist_dir = "vits/filelists/"
file_path = f"{filelist_dir}{dataset}_audio_text_all_filelist.txt"

embeddings_dict = {}
with open(file_path, "r") as file:
    for line in file:
        audiopath, sentence = line.strip().split("|")
        embeddings_dict[audiopath] = get_embedding(sentence)

# Save embeddings to a .pt file
torch.save(
    embeddings_dict, f"{filelist_dir}{dataset}_audio_bert_cls_768.pt"
)  # Replace with your desired output path
