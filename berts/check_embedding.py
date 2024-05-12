import torch
import csv

embedding_file_name = "vits/filelists/emovdb_audio_bert_cls_768"

# load embedding file
gt_embeddings_dict = torch.load(
    f"{embedding_file_name}.pt"
)  # load embedding dictionary
# print(gt_embeddings_dict)
print(len(gt_embeddings_dict))


# def count_rows_in_csv(file_path):
#     with open(file_path, 'r') as file:
#         return sum(1 for _ in file)
# file_path = 'datasets/LJSpeech-1.1/metadata.csv'
# number_of_rows = count_rows_in_csv(file_path)
# print(f"The CSV file has {number_of_rows} rows.")


# prepare writing into CSV
with open(f"{embedding_file_name}_check.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for key, tensor in gt_embeddings_dict.items():
        # get the first 5 valused in the head and tail of the tensor, respectively
        values = list(tensor[:5].numpy()) + list(tensor[-5:].numpy())
        # write into CSV
        writer.writerow([key] + values)
