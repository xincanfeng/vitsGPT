#!/bin/bash

input_directory="/data/vitsGPT/datasets/EmoV_DB_bea"
output_directory="/data/vitsGPT/datasets/EmoV_DB_bea_downsampled"

mkdir -p "$output_directory"

for input_file in "$input_directory"/*.wav; do
    filename=$(basename -- "$input_file")
    filename_noext="${filename%.*}"
    
    output_file="$output_directory/${filename_noext}.wav"
    
    # convert using ffmpeg
    ffmpeg -i "$input_file" -acodec pcm_s16le -ac 1 -ar 22050 "$output_file"
done
