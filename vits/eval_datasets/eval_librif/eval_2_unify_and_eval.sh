#!/bin/bash

# 注意不要下载假的whisper
# pip install git+https://github.com/openai/whisper.git

# 自动激活espnet虚拟环境
cd /data/espnet/egs2/libritts/tts1
. path.sh

method=$1
model=$2
step=$3

# 选择模型
# method='ori' 
# model='ljs_base'
# model='onehour_ljs_base'
# model='tenmin_ljs_base'

# method='emo'
# model='ljs_emo_add_ave'
# model='ljs_emo_add_last'
# model='ljs_emo_add_pca'
# model='ljs_emo_add_eis_word'
# model='ljs_emo_add_eis_sentence'
# model='onehour_ljs_emo_add_ave'
# model='onehour_ljs_emo_add_last'
# model='onehour_ljs_emo_add_pca'
# model='onehour_ljs_emo_add_eis_word'
# model='onehour_ljs_emo_add_eis_sentence'

# method='sem'
# model='ljs_sem_mat_phone'
# model='ljs_sem_mat_text'
# model='onehour_ljs_sem_mat_phone'
# model='onehour_ljs_sem_mat_text'
# model='tenmin_ljs_sem_mat_phone'
# model='tenmin_ljs_sem_mat_text'

# step='G_50000'
# step='G_100000'
# step='G_150000'
# step='G_200000'
# step='G_250000'
# step='G_300000'

model_step_dir="/data/vitsGPT/vits/${method}_vits/logs/${model}/${step}/"
kaldi_style_files_dir="${model_step_dir}kaldi_style_files/"

gt_wav_scp_path="${kaldi_style_files_dir}gt_wav.scp"
model_wav_scp_path="${kaldi_style_files_dir}model_wav.scp"
source_gt_wav_scp_path="${kaldi_style_files_dir}source_gt_wav.scp"
source_model_wav_scp_path="${kaldi_style_files_dir}source_model_wav.scp"

gt_audio_folder_dir='/data/vitsGPT/vits/DUMMY6/gt_test_wav/'
model_audio_folder_dir="${model_step_dir}model_test_wav/"
source_gt_audio_folder_dir='/data/vitsGPT/vits/DUMMY6/'
source_model_audio_folder_dir="${model_step_dir}source_model_test_wav/"


# stage 0. 进入模型目录，整理wav文件格式（顺便也整理一下顺序）
# # 保险起见，sort模型生成的kaldi_style文件内部的key
cd ${kaldi_style_files_dir}
sort -o gt_wav.scp gt_wav.scp
sort -o model_wav.scp model_wav.scp
sort -o text text
# 检查模型生成的kaldi_style文件的项目数是否相同
wc -l gt_wav.scp
wc -l model_wav.scp
wc -l text


# stage 1. 对model_wav降采样到和gt_wav一样的16bit（第一次保险起见，对gt_wav也作相同的操作；第二次以后就不要操作gt_wav了以免重复操作）
# 创建目标目录和临时目录
# mkdir -p ${gt_audio_folder_dir}
mkdir -p ${model_audio_folder_dir}
# 切换到源目录
cd ${source_model_audio_folder_dir}
# 遍历所有的wav文件
for wav_file in *.wav; do
    # 获取目标文件和临时文件的路径
    target_gt_wav_path="${gt_audio_folder_dir}${wav_file}"
    target_model_wav_path="${model_audio_folder_dir}${wav_file}"
    # 清空旧文件，统一进行降采样 -> 第二次以后就不要操作gt_wav了以免重复操作
    # rm -rf ${target_gt_wav_path}
    # echo "soxing..."
    # sox "${source_gt_audio_folder_dir}${wav_file}" -r 22050 -b 16 -t wav "${target_gt_wav_path}"
    # 清空旧文件，统一进行降采样
    rm -rf ${target_model_wav_path}
    echo "soxing..."
    sox "${source_model_audio_folder_dir}${wav_file}" -r 22050 -b 16 -t wav "${target_model_wav_path}"
    # 检查模型生成的 model_wav 和 gt_wav 的格式是否完全相同
    soxi "${target_gt_wav_path}"
    soxi "${target_model_wav_path}"
done


# stage 2. 重新检查文件是否存在
# python3 /data/vitsGPT/vits/eval_1_make_kaldi_style_files.py check_files_existence_in_scp ${gt_wav_scp_path} ;
# python3 /data/vitsGPT/vits/eval_1_make_kaldi_style_files.py check_files_existence_in_scp ${model_wav_scp_path} ;
# python3 /data/vitsGPT/vits/eval_1_make_kaldi_style_files.py check_files_existence_in_scp ${source_gt_wav_scp_path} ;
# python3 /data/vitsGPT/vits/eval_1_make_kaldi_style_files.py check_files_existence_in_scp ${source_model_wav_scp_path} ;


# stage 3. 在espnet中随便找一个英语recipe，利用它的代码对模型生成的wav进行评估
# . /data/espnet/egs2/libritts/tts1/eval.sh ${method} ${model} ${step}