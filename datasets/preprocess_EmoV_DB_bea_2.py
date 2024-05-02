import csv
import re

input_file = '/data/vitsGPT/datasets/EmoV_DB_bea_filtered/metadata_original.csv'
output_file = '/data/vitsGPT/datasets/EmoV_DB_bea_filtered/metadata_converted.csv'

with open(input_file, 'r') as in_file, open(output_file, 'w') as out_file:
    for line in in_file:
        match = re.match(r"\( arctic_a(\d{4}) \"(.+)\" \)", line.strip())
        if match:
            file_num, text = match.groups()
            out_file.write(f"{file_num}|{text}\n")
