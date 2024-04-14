#!/bin/bash

# 输入文件夹
input_directory="datasets/VCTK-Corpus-0.92/wav48_silence_trimmed"

# 遍历文件夹中的所有.flac文件
find "$input_directory" -type f -name '*.flac' | while read -r file
do
  # 获取文件的路径和名称，不包括扩展名
  filepath=$(dirname "$file")
  filename=$(basename "$file" .flac)
  
  # 输出文件的完整路径和名称
  output_file="$filepath/${filename}.wav"

  # 使用sox将.flac文件转换为.wav，并下采样到22050Hz
  sox "$file" -r 22050 "$output_file"

  # 输出一条消息表示文件已转换
  echo "Converted $file to $output_file"
done
