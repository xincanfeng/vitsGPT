#!/bin/bash

input_directory="vitsGPT/datasets/LibriTTS_filtered/wav_2204_131732/"
output_directory="vitsGPT/datasets/LibriTTS_filtered/wav_2204_131732_downsampled/"

# 确保输出目录存在
mkdir -p "${output_directory}"

# 遍历输入目录中的所有wav文件
for input_file in "$input_directory"/*.wav; do
    # 获取文件名，不带扩展名
    filename=$(basename -- "$input_file")
    filename_noext="${filename%.*}"
    
    # 定义输出文件路径
    output_file="$output_directory/${filename_noext}.wav"
    
    # 使用ffmpeg进行转换
    ffmpeg -i "$input_file" -acodec pcm_s16le -ac 1 -ar 22050 "$output_file"

done
