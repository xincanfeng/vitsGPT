import random
import librosa
import os


# 主要处理的文件对象
text_file = "vitsGPT/vits/filelists/ljs_audio_text_all_filelist.txt"
all_filelist = "vitsGPT/vits/filelists/ljs_audio_text_all_filelist.txt"
train24_filelist = "vitsGPT/vits/filelists/ljs_audio_text_train_filelist.txt"
train1_filelist = "vitsGPT/vits/filelists/onehour_ljs_audio_text_train_filelist.txt"
dev_filelist = "vitsGPT/vits/filelists/ljs_audio_text_val_filelist.txt"
test_filelist = "vitsGPT/vits/filelists/ljs_audio_text_test_filelist.txt"
wav_directory = "vitsGPT/vits/"


# # 2. 计算wav文件的总时长
def compute_total_duration(text_file, directory):
    # 从 filtered_text 文件中获取所有的wav文件名
    with open(text_file, "r") as f:
        lines = f.readlines()
        wav_files = [line.strip().split("|")[0] for line in lines]
    total_duration = 0.0  # 时长总和，单位为秒
    # 对于每一个wav文件，计算它的时长并累加
    for wav_file in wav_files:
        y, sr = librosa.load(os.path.join(directory, wav_file), sr=None)  # 加载wav文件
        duration = librosa.get_duration(y=y, sr=sr)  # 获取wav文件的时长
        total_duration += duration
    return total_duration


# total_duration = compute_total_duration(text_file, wav_directory)
train24_duration = compute_total_duration(train24_filelist, wav_directory)
train1_duration = compute_total_duration(train1_filelist, wav_directory)
dev_duration = compute_total_duration(dev_filelist, wav_directory)
test_duration = compute_total_duration(test_filelist, wav_directory)
# print(f"Total duration of all WAV files in filtered_text: {total_duration} seconds ({total_duration/60} minites)")
print(
    f"Train24 duration of all WAV files in filtered_text: {train24_duration} seconds ({train24_duration/60} minites)"
)
print(
    f"Train24 duration of all WAV files in filtered_text: {train1_duration} seconds ({train1_duration/60} minites)"
)
print(
    f"Dev duration of all WAV files in filtered_text: {dev_duration} seconds ({dev_duration/60} minites)"
)
print(
    f"Test duration of all WAV files in filtered_text: {test_duration} seconds ({test_duration/60} minites)"
)
