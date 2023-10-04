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

gt_wavscp="${kaldi_style_files_dir}gt_wav.scp"
model_wavscp="${kaldi_style_files_dir}model_wav.scp"
gt_text="${kaldi_style_files_dir}text"


# Redirect all output to a file
mkdir -p ${model_step_dir}
exec &> "${model_step_dir}eval_${method}_${model}_${step}.txt"


# Evaluate MCD
# ./pyscripts/utils/evaluate_mcd.py \
#     ${model_wavscp} \
#     ${gt_wavscp}
# # Evaluate log-F0 RMSE
# ./pyscripts/utils/evaluate_f0.py \
#     ${model_wavscp} \
#     ${gt_wavscp}
# If you want to calculate more precisely, limit the F0 range
# ./pyscripts/utils/evaluate_f0.py \
#     --f0min 80 \
#     --f0max 7600 \
#     ${model_wavscp} \
#     ${gt_wavscp}
# You can also use openai whisper for evaluation
./scripts/utils/evaluate_asr.sh \
    --gpu_inference true \
    --stop_stage 3 \
    --fs 22050 \
    --whisper_tag large \
    --nj 64 \
    --inference_nj 64 \
    --gt_text ${gt_text} \
    ${model_wavscp} \
    ${model_step_dir}asr_results
# # Evaluate CFSD
# ./pyscripts/utils/evaluate_cfsd.py \
#     ${model_wavscp} \
#     ${gt_wavscp}
# Evaluate SECS
# ./pyscripts/utils/evaluate_secs.py \
#     ${model_wavscp} \
#     ${gt_wavscp}

