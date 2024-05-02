import random
import librosa
import os


# 主要处理的文件对象
text_file = "vitsGPT/datasets/LibriTTS_filtered/text_2204_131732.txt"
all_filelist = "vitsGPT/vits/filelists/librif_audio_text_all_filelist.txt"
dev_filelist = "vitsGPT/vits/filelists/librif_audio_text_dev_filelist.txt"
test_filelist = "vitsGPT/vits/filelists/librif_audio_text_test_filelist.txt"
train_filelist = "vitsGPT/vits/filelists/librif_audio_text_train_filelist.txt"
wav_directory = "vitsGPT/vits/"


# 1. 划分数据集
def split_dataset(text_file):
    with open(text_file, 'r') as f:
        lines = f.readlines()
    # 打乱行的顺序
    random.shuffle(lines)
    # 格式转换函数
    def convert_format(line):
        key, value = line.strip().split(" ", 1)
        return f"DUMMY6/{key}.wav|{value}\n"
    # 提取dev, test和train的数据，并进行格式转换
    all_lines = [convert_format(line) for line in lines[:]]
    dev_lines = [convert_format(line) for line in lines[:30]]
    test_lines = [convert_format(line) for line in lines[30:60]]
    train_lines = [convert_format(line) for line in lines[60:]]
    # 保存到新的文件
    with open(all_filelist, "w") as f:
        f.writelines(all_lines)
    with open(dev_filelist, "w") as f:
        f.writelines(dev_lines)
    with open(test_filelist, "w") as f:
        f.writelines(test_lines)
    with open(train_filelist, "w") as f:
        f.writelines(train_lines)
# 使用方法
split_dataset(text_file)


# # 2. 计算wav文件的总时长
def compute_total_duration(text_file, directory):
    # 从 filtered_text 文件中获取所有的wav文件名
    with open(text_file, 'r') as f:
        lines = f.readlines()
        wav_files = [line.strip().split('|')[0] for line in lines]
    total_duration = 0.0  # 时长总和，单位为秒
    # 对于每一个wav文件，计算它的时长并累加
    for wav_file in wav_files:
        y, sr = librosa.load(os.path.join(directory, wav_file), sr=None)  # 加载wav文件
        duration = librosa.get_duration(y=y, sr=sr)  # 获取wav文件的时长
        total_duration += duration
    return total_duration
# total_duration = compute_total_duration(text_file, wav_directory)
train_duration = compute_total_duration(train_filelist, wav_directory)
dev_duration = compute_total_duration(dev_filelist, wav_directory)
test_duration = compute_total_duration(test_filelist, wav_directory)
# print(f"Total duration of all WAV files in filtered_text: {total_duration} seconds ({total_duration/60} minites)")
print(f"Train duration of all WAV files in filtered_text: {train_duration} seconds ({train_duration/60} minites)")
print(f"Dev duration of all WAV files in filtered_text: {dev_duration} seconds ({dev_duration/60} minites)")
print(f"Test duration of all WAV files in filtered_text: {test_duration} seconds ({test_duration/60} minites)")