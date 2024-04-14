import os
import random
import pandas as pd
import shutil
import uuid


# Provided file folder addresses (replace with actual paths)
common_dir = "/data/vitsGPT/vits/"
folders = [
    f"{common_dir}DUMMY1/gt_test_wav/",
    f"{common_dir}ori_vits/logs/ljs_base/G_90000/model_test_wav/",
    f"{common_dir}emo_vits/logs/ljs_emo_add_ave/G_100000/model_test_wav/",
    f"{common_dir}emo_vits/logs/ljs_emo_add_bert_cls/G_100000/model_test_wav/",
    f"{common_dir}sem_vits/logs/ljs_sem_mat_text/G_100000/model_test_wav/",
    f"{common_dir}sem_vits/logs/ljs_sem_mat_phone/G_100000/model_test_wav/",
    f"{common_dir}sem_vits/logs/ljs_sem_mat_bert_text/G_100000/model_test_wav/",
    f"{common_dir}sem_vits/logs/ljs_sem_mat_bert_phone/G_100000/model_test_wav/",
]

num_extracted_files = 150

# 0. Create human_mos directory if it doesn't exist
human_mos_dir = os.path.join(common_dir, "human_evaluation_ljs/human_mos_wavs")
if not os.path.exists(human_mos_dir):
    os.makedirs(human_mos_dir)
    
# 1. Check if every folder has the same number and names of files
reference_files = set(os.listdir(folders[0]))
consistent = all(set(os.listdir(folder)) == reference_files for folder in folders)
if not consistent:
    raise ValueError("Folders do not have consistent file names or counts.")

# 2. Extract, rename, and copy files from each folder
sample_files = random.sample(list(reference_files), num_extracted_files)

# Generate a shuffled list of numbers for renaming
numbers = list(range(len(sample_files) * len(folders)))
random.shuffle(numbers)

renamed_files_paths = {}
for orig in sample_files:
    for folder in folders:
        # new_name = f"{len(renamed_files_paths)}.wav"
        
        # Use UUID to generate a random file name
        # new_name = f"{uuid.uuid4().hex}.wav"
        
        # Use shuffled number for renaming
        new_name = f"{numbers.pop()}.wav"
        
        src = os.path.join(folder, orig)
        dst = os.path.join(human_mos_dir, new_name)
        shutil.copy(src, dst)
        renamed_files_paths[src] = new_name
        
df_named_score = pd.DataFrame({
    "original_file_path": list(renamed_files_paths.keys()),
    "innominated_file_path": list(renamed_files_paths.values())
})

# 3. Create innominated_score.xlsx
df_innominated_score = pd.DataFrame({
    "file_name": list(renamed_files_paths.values()),
    "MOS_score": "",
    "Emo/Not": "",
})

# Save to excel
df_combined = pd.concat([df_innominated_score])
df_named_score.to_excel("named_score.xlsx", index=False)
df_combined.to_excel("innominated_score.xlsx", index=False)

"Files created and copied successfully."





# 把输出的excel表格整理一下，按顺序写音频编号，方便annotator录入
# Load the original innominated_score.xlsx
df_to_sort = pd.read_excel("innominated_score.xlsx")
# Extract the number from the filename for each row
df_to_sort['sort_key'] = df_to_sort['file_name'].str.extract('(\d+)').astype(int)
# Sort the dataframe by this new column
df_sorted = df_to_sort.sort_values(by="sort_key")
# Drop the 'sort_key' column as it's no longer needed
df_sorted = df_sorted.drop(columns=['sort_key'])
# Save the sorted dataframe as scoring.xlsx
df_sorted.to_excel("scoring.xlsx", index=False)
print("File scoring.xlsx created and sorted successfully.")

