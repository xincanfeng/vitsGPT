#!/bin/bash

# input folder
input_directory="datasets/VCTK-Corpus-0.92/wav48_silence_trimmed"

# check all .flac file
find "$input_directory" -type f -name '*.flac' | while read -r file
do
  # get file path
  filepath=$(dirname "$file")
  filename=$(basename "$file" .flac)
  
  # output file path
  output_file="$filepath/${filename}.wav"

  # use sox to convert .flac into .wav, and downsampling to 22050Hz
  sox "$file" -r 22050 "$output_file"

  # converted
  echo "Converted $file to $output_file"
done
