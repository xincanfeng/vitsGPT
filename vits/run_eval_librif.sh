#!/bin/bash

# 选择模型
# method='ori' 
# model='librif_base_pretrained'

method='emo'
model='librif_emo_add_ave_pretrained'
# model='librif_emo_add_last_pretrained'
# model='librif_emo_add_pca_pretrained'
# model='librif_emo_add_eis_word_pretrained'
# model='librif_emo_add_eis_sentence_pretrained'
# model='librif_emo_add_bert_cls_pretrained'

# method='sem'
# model='librif_sem_mat_phone_pretrained'
# model='librif_sem_mat_text_pretrained'
# model='librif_sem_mat_bert_phone_pretrained'
# model='librif_sem_mat_bert_text_pretrained'

step='G_50000'
# step='G_100000'
# step='G_150000'
# step='G_200000'
# step='G_250000'
# step='G_300000'


# 1. 执行 Python 脚本，给model_wav重命名，并创建相关scp文件
python3 /data/vitsGPT/vits/eval_datasets/eval_librif/eval_1_make_kaldi_style_files.py ${method} ${model} ${step}

# 2. 执行 Bash 脚本，对model_wav（和gt_wav）降采样
. /data/vitsGPT/vits/eval_datasets/eval_librif/eval_2_unify_and_eval_init.sh ${method} ${model} ${step} # 第一次的话执行这个，后面就执行下面的了
# . /data/vitsGPT/vits/eval_librif/eval_2_unify_and_eval.sh ${method} ${model} ${step}

# 3. 在espnet中随便找一个英语recipe，利用它的代码对模型生成的wav进行客观指标MCD，ASR，F0等的评估
CUDA_VISIBLE_DEVICES=0 . /data/espnet/egs2/libritts/tts1/eval.sh ${method} ${model} ${step} 
# 可选：在后台运行
# CUDA_VISIBLE_DEVICES=0 nohup /data/espnet/egs2/libritts/tts1/eval.sh ${method} ${model} ${step} > eval.log 2>&1 & 

# 4. 用SpeechMOS评估UTMOS
CUDA_VISIBLE_DEVICES=0 python3 /data/vitsGPT/vits/eval_datasets/eval_librif/eval_3_mos.py ${method} ${model} ${step}