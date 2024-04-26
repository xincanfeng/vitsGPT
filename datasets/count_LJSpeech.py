import random
import librosa
import os


# configure your file path
text_file = "/data/vitsGPT/vits/filelists/ljs_audio_text_all_filelist.txt"
all_filelist = "/data/vitsGPT/vits/filelists/ljs_audio_text_all_filelist.txt"
train24_filelist = "/data/vitsGPT/vits/filelists/ljs_audio_text_train_filelist.txt"
train1_filelist = "/data/vitsGPT/vits/filelists/onehour_ljs_audio_text_train_filelist.txt"
dev_filelist = "/data/vitsGPT/vits/filelists/ljs_audio_text_val_filelist.txt"
test_filelist = "/data/vitsGPT/vits/filelists/ljs_audio_text_test_filelist.txt"
wav_directory = "/data/vitsGPT/vits/"


# # 2. calculate the total duration of wav
def compute_total_duration(text_file, directory):
    # get all wav name from filtered_text file
    with open(text_file, 'r') as f:
        lines = f.readlines()
        wav_files = [line.strip().split('|')[0] for line in lines]
    total_duration = 0.0  # total duration (second)
    # sum all wav duration
    for wav_file in wav_files:
        y, sr = librosa.load(os.path.join(directory, wav_file), sr=None)  # load wav file
        duration = librosa.get_duration(y=y, sr=sr)  # get wav duration
        total_duration += duration
    return total_duration
# total_duration = compute_total_duration(text_file, wav_directory)
train24_duration = compute_total_duration(train24_filelist, wav_directory)
train1_duration = compute_total_duration(train1_filelist, wav_directory)
dev_duration = compute_total_duration(dev_filelist, wav_directory)
test_duration = compute_total_duration(test_filelist, wav_directory)
# print(f"Total duration of all WAV files in filtered_text: {total_duration} seconds ({total_duration/60} minites)")
print(f"Train24 duration of all WAV files in filtered_text: {train24_duration} seconds ({train24_duration/60} minites)")
print(f"Train24 duration of all WAV files in filtered_text: {train1_duration} seconds ({train1_duration/60} minites)")
print(f"Dev duration of all WAV files in filtered_text: {dev_duration} seconds ({dev_duration/60} minites)")
print(f"Test duration of all WAV files in filtered_text: {test_duration} seconds ({test_duration/60} minites)")