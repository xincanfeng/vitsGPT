import os
import random
import pandas as pd
import shutil
import uuid


# Provided file folder addresses (replace with actual paths)
common_dir = "/data/vitsGPT/vits/"
folder_paths = [
    f"{common_dir}DUMMY5/gt_test_wav/",
    f"{common_dir}ori_vits/logs/emovdb_base_pretrained16/G_150000/model_test_wav/",
    f"{common_dir}emo_vits/logs/emovdb_emo_add_ave_pretrained16/G_150000/model_test_wav/",
    f"{common_dir}emo_vits/logs/emovdb_emo_add_bert_cls_pretrained16/G_150000/model_test_wav/",
    f"{common_dir}sem_vits/logs/emovdb_sem_mat_text_pretrained16/G_150000/model_test_wav/",
    f"{common_dir}sem_vits/logs/emovdb_sem_mat_phone_pretrained16/G_150000/model_test_wav/",
    f"{common_dir}sem_vits/logs/emovdb_sem_mat_bert_text_pretrained16/G_150000/model_test_wav/",
    f"{common_dir}sem_vits/logs/emovdb_sem_mat_bert_phone_pretrained16/G_150000/model_test_wav/",
]
# Identifiers for each folder
folder_identifiers = [
    "gt",
    "ori",
    "emo_ave",
    "emo_bert_cls",
    "sem_text",
    "sem_phone",
    "sem_bert_text",
    "sem_bert_phone"
]
# Map folders to their identifiers
folders = dict(zip(folder_paths, folder_identifiers))
# Provided file names to be extracted from each folder

# Extract those where sem_text perform good
# files_to_extract = [
#     "amused_46-56_0056.wav",
#     "amused_57-84_0068.wav",
#     "amused_57-84_0076.wav",
#     "amused_85-112_0094.wav",
#     "angry_29-56_0047.wav",
#     "disgustededed_113-140_0114.wav",
#     "disgustededed_85-112_0086.wav",
#     "neutral_57-84_0057.wav",
#     "neutral_57-84_0079.wav",
#     "sleepy_29-56_0049.wav",
# ]

# Extract all files from the first folder
# files_to_extract = os.listdir(folder_paths[0])

# Extract those where sem_bert_text perform good
# files_to_extract = [
#     "amused_85-112_0099.wav",
#     "amused_85-112_0107.wav",
#     "angry_29-56_0047.wav",
#     "disgustededed_85-112_0089.wav",
#     "disgustededed_85-112_0109.wav",
#     "neutral_29-56_0048.wav",
#     "neutral_57-84_0063.wav",
#     "neutral_85-112_0088.wav",
#     "neutral_85-112_0095.wav",
# ]

# Extract those where emo_ave perform good
files_to_extract = [
    "amused_57-84_0070.wav",
    "amused_85-112_0094.wav",
    "angry_29-56_0047.wav",
    "angry_57-84_0075.wav",
    "disgustededed_85-112_0109.wav",
    "disgustededed_113-140_0114.wav",
    "neutral_57-84_0069.wav",
    "neutral_57-84_0079.wav",
    "neutral_85-112_0088.wav",
    "neutral_85-112_0110.wav",
    "neutral_85-112_0112.wav",
]

test_text_file = f"{common_dir}filelists/emovdb_audio_text_test_filelist.txt"

# 0. Create human_mos directory if it doesn't exist
human_mos_dir = os.path.join(common_dir, "human_evaluation_emovdb_selected/human_mos_wavs_selected_emo_ave")
if not os.path.exists(human_mos_dir):
    os.makedirs(human_mos_dir)

# 定义一个找到音频对应text并另存为文件的函数
def find_and_save_value_to_file(file_path, key):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            k, v = line.strip().split('|')
            if k == f"DUMMY5/{key}.wav":
                with open(f"{human_mos_dir}/{key}.txt", 'w') as out_file:
                    out_file.write(line)
                print(f"Text for key '{key}' saved in '{key}.txt'")
                return
        print(f"Key '{key}' not found in the file.")

# 1. Check if every folder has the same number and names of files
reference_files = set(os.listdir(next(iter(folders))))
consistent = all(set(os.listdir(folder)) == reference_files for folder in folders.keys())
if not consistent:
    raise ValueError("Folders do not have consistent file names or counts.")

# 2. Extract, rename, and copy files from each folder
renamed_files_paths = {}
for orig in files_to_extract:
    wavfile_name, wavfile_extension = os.path.splitext(orig)
    find_and_save_value_to_file(test_text_file, wavfile_name)
    for folder, identifier in folders.items():
        new_name = f"{wavfile_name}-{identifier}.wav"
        src = os.path.join(folder, orig)
        
        # Check if the file exists before trying to copy
        if not os.path.exists(src):
            print(f"File {src} does not exist!")
            continue
            
        dst = os.path.join(human_mos_dir, new_name)
        shutil.copy(src, dst)
        renamed_files_paths[src] = new_name
        print(f"File {src} copied to {dst}")

# Create dataframes and save them to Excel
df_named_score = pd.DataFrame({
    "original_file_path": list(renamed_files_paths.keys()),
    "innominated_file_path": list(renamed_files_paths.values())
})
df_innominated_score = pd.DataFrame({
    "file_name": list(renamed_files_paths.values()),
    "MOS_score": "",
})

# Save to excel
df_named_score.to_excel("emovdb_named_mos_score.xlsx", index=False)
df_innominated_score.to_excel("emovdb_innominated_mos_score.xlsx", index=False)

print("Files created and copied successfully.")

# Organize the output Excel file for ease of annotator entry
# Load the original innominated_score.xlsx
df_to_sort = pd.read_excel("emovdb_innominated_mos_score.xlsx")
# Extract the number from the filename for each row
df_to_sort['sort_key'] = df_to_sort['file_name'].str.extract('(\d+)').astype(int)
# Sort the dataframe by this new column
df_sorted = df_to_sort.sort_values(by="sort_key")
# Drop the 'sort_key' column as it's no longer needed
df_sorted = df_sorted.drop(columns=['sort_key'])
# Save the sorted dataframe as scoring.xlsx
df_sorted.to_excel("emovdb_mos_scoring.xlsx", index=False)
print("File scoring.xlsx created and sorted successfully.")
