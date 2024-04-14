import os
import shutil
import string
from collections import defaultdict
import sys

# 使用 sys.argv 来获取命令行参数
method = sys.argv[1]
model = sys.argv[2]
step = sys.argv[3]

# # 定义源文件和目标文件的路径
# method = 'ori' 
# model = 'ljs_base'
# model = 'onehour_ljs_base'
# model = 'tenmin_ljs_base'

# method = 'emo'
# model = 'ljs_emo_add_ave'
# model = 'ljs_emo_add_last'
# model = 'ljs_emo_add_pca'
# model = 'ljs_emo_add_eis_word'
# model = 'ljs_emo_add_eis_sentence'
# model = 'onehour_ljs_emo_add_ave'
# model = 'onehour_ljs_emo_add_last'
# model = 'onehour_ljs_emo_add_pca'
# model = 'onehour_ljs_emo_add_eis_word'
# model = 'onehour_ljs_emo_add_eis_sentence'

# method = 'sem'
# model = 'ljs_sem_mat_phone'
# model = 'ljs_sem_mat_text'
# model = 'onehour_ljs_sem_mat_phone'
# model = 'onehour_ljs_sem_mat_text'
# model = 'tenmin_ljs_sem_mat_phone'
# model = 'tenmin_ljs_sem_mat_text'

# step = 'G_50000'
# step = 'G_100000'
# step = 'G_150000'
# step = 'G_200000'
# step = 'G_250000'
# step = 'G_300000'

model_step_dir = f"/data/vitsGPT/vits/{method}_vits/logs/{model}/{step}/"
kaldi_style_files_dir = f"{model_step_dir}kaldi_style_files/"
source_text_path = "/data/vitsGPT/vits/filelists/ljs_audio_text_test_filelist.txt"

gt_audio_folder_dir = '/data/vitsGPT/vits/DUMMY1/gt_test_wav/'
model_audio_folder_dir = f"{model_step_dir}model_test_wav/"  # 包含要重命名的文件的文件夹的路径
source_gt_audio_folder_dir = '/data/vitsGPT/vits/DUMMY1/'
source_model_audio_folder_dir = f'{model_step_dir}source_model_test_wav/'

text_file_path = f"{kaldi_style_files_dir}text"
gt_wav_scp_path = f"{kaldi_style_files_dir}gt_wav.scp"
model_wav_scp_path = f"{kaldi_style_files_dir}model_wav.scp"
source_gt_wav_scp_path = f"{kaldi_style_files_dir}source_gt_wav.scp"
source_model_wav_scp_path = f"{kaldi_style_files_dir}source_model_wav.scp"


# stage 0. 清除相关文件夹下的旧有文件
def clear_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
# 使用方法
if os.path.exists(kaldi_style_files_dir) and os.path.isdir(kaldi_style_files_dir):
    clear_directory(kaldi_style_files_dir)


# stage 1. 制作text文件
# 确保目标文件的父目录存在
destination_dir = os.path.dirname(text_file_path)
os.makedirs(destination_dir, exist_ok=True)
# 或者使用 shutil.copy2() 复制文件，它会尝试保留文件元数据
shutil.copy2(source_text_path, f"{text_file_path}_bk")
# 读取源文件的内容
with open(source_text_path, 'r') as source_file:
    lines = source_file.readlines()
# 处理每一行并写入目标文件
with open(text_file_path, 'w') as dest_file:
    for line in lines:
        # 分割 key 和 value
        key, value = line.strip().split('|', 1)
        # 删除 "DUMMY1/" 字段
        key = key.replace('DUMMY1/', '')
        # 转换 value 为大写
        value = value.upper()
        # 删除所有标点符号
        value = value.translate(str.maketrans('', '', string.punctuation))
        # 重新组合 key 和 value，并写入目标文件
        new_line = f"{key} {value}\n"
        dest_file.write(new_line)


# stage 2. 准备工作：为顺次生成的test wav根据utt重命名（如果已经重命名过再跑代码也没关系）
def rename_wav_files(text_file_path, source_model_audio_folder_dir):
    # 读取文本文件并获取每一行的 key
    with open(text_file_path, 'r') as file:
        keys = [line.split(' ')[0].replace('DUMMY1/', '') for line in file]
    for index, key in enumerate(keys):
        # 构造旧的文件名和新的文件名
        old_filename = os.path.join(source_model_audio_folder_dir, f'output_{method}_{index}.wav')
        new_filename = os.path.join(source_model_audio_folder_dir, f'{key}')
        # 检查旧的文件是否存在
        if os.path.exists(old_filename):
            # 重命名文件
            os.rename(old_filename, new_filename)
        else:
            print(f"File {old_filename} does not exist.")
# 使用方法
rename_wav_files(text_file_path, source_model_audio_folder_dir)


# stage 3. 制作wav.scp文件
def audio_to_wav(text_file_path, audio_folder_dir, output_file_path):
    # 确保输出文件的父目录存在
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(text_file_path, 'r') as text_file:
        # 读取文本文件的每一行，并将每一行的 key 分割出来
        keys = [line.strip().split(' ')[0] for line in text_file]
    with open(output_file_path, 'w') as output_file:
        for key in keys:
            # 构造 wav 文件的路径
            # 删除 'DUMMY1/' 以构造文件名
            wav_filename = key.replace('DUMMY1/', '')
            wav_path = os.path.join(audio_folder_dir, wav_filename)
            # 写入新文件
            output_file.write(f"{key} {wav_path}\n")
    print(f"wav scp file saved to {output_file_path}")
# 使用方法
audio_to_wav(text_file_path, gt_audio_folder_dir, gt_wav_scp_path)
audio_to_wav(text_file_path, model_audio_folder_dir, model_wav_scp_path)
audio_to_wav(text_file_path, source_gt_audio_folder_dir, source_gt_wav_scp_path)
audio_to_wav(text_file_path, source_model_audio_folder_dir, source_model_wav_scp_path)


# stage 4. 根据scp文件，检查gt_wav和model_wav文件是否存在
def check_files_existence_in_scp(scp_file_path):
    # 创建一个空列表来保存不存在的文件路径
    missing_files = []
    # 创建一个字典来记录每个文件夹中缺失文件的数量
    missing_counts_per_directory = defaultdict(int)
    # 创建一个字典来记录每个文件夹中的文件总数
    total_counts_per_directory = defaultdict(int)
    with open(scp_file_path, 'r') as scp_file:
        for line in scp_file:
            # 分割每一行以获取文件路径（value 部分）
            key, file_path = line.strip().split(' ')
            # 更新文件夹的文件总数
            directory = os.path.dirname(file_path)
            total_counts_per_directory[directory] += 1
            # 检查文件是否存在
            if not os.path.exists(file_path):
                missing_files.append(file_path)
                # 更新缺失文件的数量
                missing_counts_per_directory[directory] += 1
    # 如果有缺失的文件，打印它们的路径
    if missing_files:
        print(f"Missing files: {len(missing_files)}")
        # for missing_file in missing_files:
        #     print(f"Missing: {missing_file}")
    else:
        print(f"{scp_file_path}: All files exist.")
    # 检查是否有完全缺失的文件夹
    for directory, missing_count in missing_counts_per_directory.items():
        total_count = total_counts_per_directory[directory]
        if missing_count == total_count:
            print(f"{directory}: All files in this directory are missing.")
# 使用方法
check_files_existence_in_scp(gt_wav_scp_path)
check_files_existence_in_scp(model_wav_scp_path)
check_files_existence_in_scp(source_gt_wav_scp_path)
check_files_existence_in_scp(source_model_wav_scp_path)