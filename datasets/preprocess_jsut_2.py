import MeCab
import csv

# 初始化MeCab
m = MeCab.Tagger("-r /etc/mecabrc -Oyomi")

# 增加平假名注释
def kanji_to_hiragana(text):
    return m.parse(text).strip()

with open('jsut-1.1/jsut-1.1-processed/all_tex/combined.csv', 'r', encoding='utf-8') as infile, open('jsut-1.1/jsut-1.1-processed/all_tex/combined_hiragana.csv', 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile, delimiter='|')
    writer = csv.writer(outfile, delimiter='|')

    for row in reader:
        original_text = row[1]
        hiragana = kanji_to_hiragana(original_text)
        writer.writerow([row[0], original_text, hiragana])