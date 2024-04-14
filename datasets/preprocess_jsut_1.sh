# 复制所需的文件
# Step 1: 创建新的文件夹结构
mkdir -p jsut-1.1/jsut-1.1-processed/all_tex
mkdir -p jsut-1.1/jsut-1.1-processed/all_wavs
# Step 2: 复制wav文件和transcript_utf8.txt文件到新的文件夹中
for folder in basic5000 countersuffix26 loanword128 onomatopee300 precedent130 repeat500 travel1000 utparaphrase512 voiceactress100; do
    # 复制wav文件夹
    cp -r jsut-1.1/${folder}/wav/* jsut-1.1/jsut-1.1-processed/all_wavs/
    # 复制transcript_utf8.txt文件
    cp jsut-1.1/${folder}/transcript_utf8.txt jsut-1.1/jsut-1.1-processed/all_tex/${folder}_transcript_utf8.txt
done

# 检查wav文件的采样率，如果高于22050Hz，则进行downsample
TARGET_DIR="jsut-1.1/jsut-1.1-processed/all_wavs"  # wav文件夹路径
TARGET_RATE=22050
for file in "$TARGET_DIR"/*.wav; do
    CURRENT_RATE=$(soxi -r "$file")
    if (( CURRENT_RATE > TARGET_RATE )); then
        echo "Downsampling $file from $CURRENT_RATE to $TARGET_RATE"
        sox "$file" -r $TARGET_RATE "${file%.wav}_temp.wav"
        mv "${file%.wav}_temp.wav" "$file"
    fi
done

# 合并文本文件
# 切换到all_tex文件夹
cd jsut-1.1/jsut-1.1-processed/all_tex
# 合并所有txt文件并将其格式修改为csv格式
cat *.txt | sed 's/:/|/' > combined.csv

# 处理文本内容：增加平假名注释
python preprocess_2_jsut.py
