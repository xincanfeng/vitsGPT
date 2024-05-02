import re
import os
import shutil
import librosa
import random
import math


# 1. 清理llama输出的文件，并核对metadata.csv文件，最终生成audio|emotion|text文件
i = 3
# 从 text 文件中的每行提取目标情感词汇
def extract_emotion(line):
    emotions = ['Amused', 'Angry', 'Neutral', 'Disgusted', 'Sleepy']
    for emotion in emotions:
        if emotion in line:
            return emotion
    return ''
# 读取 meta.csv 文件并创建一个句子到编号的映射
with open('/data/vitsGPT/datasets/EmoV_DB_bea_filtered/metadata.csv', 'r') as meta_file:
    meta_dict = {line.split('|')[1].strip(): line.split('|')[0] for line in meta_file.readlines()}
# 处理 text 文件
with open(f'/data/vitsGPT/datasets/EmoV_DB_bea_filtered/emovdb-bea_audio_emo_{i}.txt', 'r') as text_file:
    lines = text_file.readlines()
with open(f'/data/vitsGPT/datasets/EmoV_DB_bea_filtered/audio_llama-emo_text_{i}.txt', 'w') as output_file:
    for line in lines:
        pre, post = line.split('|')
        emotion = extract_emotion(pre)
        if post.strip() in meta_dict:
            new_line = f"{meta_dict[post.strip()]}|{emotion}|{post}"
            output_file.write(new_line)
        else:
            print(f"Warning: '{post.strip()}' not found in meta.csv!")


# 2. 核对每个生成audio|emotion|text文件的感情标签是否一致，并输出不一致的audio
# 所有audio|emotion|text文件所在目录
directory_path = "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/audio_llama-emo_text/"
# 从文件中读取内容
def read_file_content(filepath):
    with open(filepath, 'r') as file:
        return file.readlines()
# 获取所有文件内容
all_files_content = [read_file_content(os.path.join(directory_path, filename)) for filename in os.listdir(directory_path) if filename.endswith(".txt")]
inconsistency_found = False
# 核对第一列和第三列的内容是否完全一致
for i in range(len(all_files_content[0])):
    first_column_values = [content[i].split('|')[0] for content in all_files_content]
    third_column_values = [content[i].split('|')[2] for content in all_files_content]
    if len(set(first_column_values)) != 1 or len(set(third_column_values)) != 1:
        print(f"Inconsistency detected in line {i+1}: {first_column_values[0]}")
        inconsistency_found = True
        break
# 核对第二列的内容是否完全一致
for i in range(len(all_files_content[0])):
    second_column_values = [content[i].split('|')[1] for content in all_files_content]
    if len(set(second_column_values)) > 1:
        print(f"Inconsistency detected in emotions for line {first_column_values[0]}: {second_column_values}")
        inconsistency_found = True
        break
if not inconsistency_found:
    print("All contents are consistent across files!")


# 3. 根据文本文件中的编号和情感，如果可以找到目录中唯一匹配的wav文件，就按照“编号|情感|wav文件名“的形式记录在filtered.text文件中；
# 注意，如果找不到匹配的wav文件，或者匹配的wav文件不止一个，请记录这些编号、打印出来。
def read_text_file(filepath):
    with open(filepath, 'r') as file:
        return file.readlines()
# 获取目录中所有wav文件
def get_all_wav_files(directory_path):
    return [f for f in os.listdir(directory_path) if f.endswith(".wav")]
# 标准化wav文件名
def normalize_wav_filename(filename):
    normalized_name = filename.lower()
    normalized_name = normalized_name.replace("sleepiness", "sleepy")
    normalized_name = normalized_name.replace("disgust", "disgusted")
    normalized_name = normalized_name.replace("anger", "angry")
    return normalized_name
def filter_and_copy_wavs(text_file, filtered_text_file, source_directory, destination_directory):
    # 读取文本文件内容
    lines = read_text_file(text_file)
    # Step 1: 标准化source_directory内的所有文件名
    all_wav_files = get_all_wav_files(source_directory)
    for wav_file in all_wav_files:
        normalized_name = normalize_wav_filename(wav_file)
        os.rename(os.path.join(source_directory, wav_file), os.path.join(source_directory, normalized_name))
    unmatched_ids = []
    with open(filtered_text_file, "w") as out_file:
        for line in lines:
            id, emotion, sentence = line.strip().split('|')
            if not emotion:  # 检查情感是否为空
                unmatched_ids.append(id)
                continue
            matching_files = [f for f in get_all_wav_files(source_directory) if f.endswith(f"_{id}.wav") and emotion.lower() in f]
            # 检查是否有唯一匹配的wav文件
            if len(matching_files) == 1:
                shutil.copy(os.path.join(source_directory, matching_files[0]), os.path.join(destination_directory, matching_files[0]))
                out_file.write(f"{id}|{emotion}|{matching_files[0]}|{sentence}\n")
            else:
                unmatched_ids.append(id)
    if unmatched_ids:
        print(f"Unmatched or multiple matches found for IDs: {', '.join(unmatched_ids)}")
# 使用方法
text_file = "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/audio_llama-emo_text.txt"
filtered_text_file = "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/audio_llama-emo_wav_filtered.txt"
source_directory = "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/wavs_bk"
destination_directory = "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/wavs_filtered"
filter_and_copy_wavs(text_file, filtered_text_file, source_directory, destination_directory)


# # 4. 计算过滤出来的所有符合文本的情感表达的wav文件的总时长
def compute_total_duration(filtered_text_file, directory):
    # 从 filtered_text 文件中获取所有的wav文件名
    with open(filtered_text_file, 'r') as f:
        lines = f.readlines()
        wav_files = [line.strip().split('|')[-2] for line in lines]
    total_duration = 0.0  # 时长总和，单位为秒
    # 对于每一个wav文件，计算它的时长并累加
    for wav_file in wav_files:
        y, sr = librosa.load(os.path.join(directory, wav_file), sr=None)  # 加载wav文件
        duration = librosa.get_duration(y=y, sr=sr)  # 获取wav文件的时长
        total_duration += duration
    return total_duration
filtered_text_file = "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/audio_llama-emo_wav_filtered.txt"
train_file = "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/train.txt"
dev_file = "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/dev.txt"
test_file = "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/test.txt"
directory = "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/wavs_filtered"
total_duration = compute_total_duration(filtered_text_file, directory)
train_duration = compute_total_duration(train_file, directory)
dev_duration = compute_total_duration(dev_file, directory)
test_duration = compute_total_duration(test_file, directory)
print(f"Total duration of all WAV files in filtered_text: {total_duration} seconds ({total_duration/60} minites)")
print(f"Train duration of all WAV files in filtered_text: {train_duration} seconds ({train_duration/60} minites)")
print(f"Dev duration of all WAV files in filtered_text: {dev_duration} seconds ({dev_duration/60} minites)")
print(f"Test duration of all WAV files in filtered_text: {test_duration} seconds ({test_duration/60} minites)")


# 5. 把过滤出来的所有符合文本的情感表达的数据集按照总体情感比例分成train, dev和test
def split_dataset_maintain_ratio(filtered_text_file, train_file, dev_file, test_file, dev_size=50, test_size=50):
    # 读取filtered_text文件
    with open(filtered_text_file, 'r') as f:
        lines = f.readlines()
    # 按情感分类记录
    emotion_dict = {}
    for line in lines:
        _, emotion, _, _ = line.strip().split('|')
        emotion_dict.setdefault(emotion, []).append(line)
    dev_set = []
    test_set = []
    total_size = len(lines)
    # 为每种情感选择适当数量的记录
    for emotion, records in emotion_dict.items():
        ratio = len(records) / total_size
        num_dev = math.ceil(dev_size * ratio)
        num_test = math.ceil(test_size * ratio)
        dev_set.extend(records[:num_dev])
        test_set.extend(records[num_dev:num_dev+num_test])
        # 从原记录中删除已选中的记录
        del records[:num_dev+num_test]
    # 剩余的记录作为train集
    train_set = [line for sublist in emotion_dict.values() for line in sublist]
    # 将结果写入相应的文件
    with open(train_file, 'w') as f:
        f.writelines(train_set)
    with open(dev_file, 'w') as f:
        f.writelines(dev_set)
    with open(test_file, 'w') as f:
        f.writelines(test_set)
filtered_text_file = "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/audio_llama-emo_wav_filtered.txt"
train_file = "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/train.txt"
dev_file = "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/dev.txt"
test_file = "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/test.txt"
split_dataset_maintain_ratio(filtered_text_file, train_file, dev_file, test_file)


# 6. 函数检查一下原来的文件和生成的三个子文件的每个情感比例
def emotion_distribution_in_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    emotion_dict = {}
    for line in lines:
        _, emotion, _, _ = line.strip().split('|')
        emotion_dict[emotion] = emotion_dict.get(emotion, 0) + 1
    total = len(lines)
    distribution = {emotion: count/total for emotion, count in emotion_dict.items()}
    return distribution
def print_emotion_distribution(original_file, train_file, dev_file, test_file):
    original_dist = emotion_distribution_in_file(original_file)
    train_dist = emotion_distribution_in_file(train_file)
    dev_dist = emotion_distribution_in_file(dev_file)
    test_dist = emotion_distribution_in_file(test_file)
    print("Emotion distribution in Original File:")
    for emotion, ratio in original_dist.items():
        print(f"{emotion}: {ratio*100:.2f}%")
    print("\nEmotion distribution in Train File:")
    for emotion, ratio in train_dist.items():
        print(f"{emotion}: {ratio*100:.2f}%")
    print("\nEmotion distribution in Dev File:")
    for emotion, ratio in dev_dist.items():
        print(f"{emotion}: {ratio*100:.2f}%")
    print("\nEmotion distribution in Test File:")
    for emotion, ratio in test_dist.items():
        print(f"{emotion}: {ratio*100:.2f}%")
# 使用方法
original_file = "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/audio_llama-emo_wav_filtered.txt"
train_file = "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/train.txt"
dev_file = "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/dev.txt"
test_file = "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/test.txt"
print_emotion_distribution(original_file, train_file, dev_file, test_file)


# 7. clean一下重复信息，把数据集中的编号和情感去掉，并加上vits中的地址信息(DUMMY5)，生成“DUMMY5/wav|text“文件
def clean_and_save_data(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    cleaned_lines = []
    for line in lines:
        _, _, wav, text = line.strip().split('|')
        cleaned_lines.append(f"DUMMY5/{wav}|{text}")
    with open(output_file, 'w') as f:
        f.write('\n'.join(cleaned_lines))
# 使用方法
input_files = [
    "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/train.txt",
    "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/dev.txt",
    "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/test.txt"
]
output_files = [
    "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/train_cleaned.txt",
    "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/dev_cleaned.txt",
    "/data/vitsGPT/datasets/EmoV_DB_bea_filtered/test_cleaned.txt"
]
for input_file, output_file in zip(input_files, output_files):
    clean_and_save_data(input_file, output_file)
