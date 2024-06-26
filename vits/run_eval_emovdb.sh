#!/bin/bash

# choose model
method='ori' 
model='emovdb_base_pretrained16'

# method='emo'
# model='emovdb_emo_add_ave_pretrained16'
# model='emovdb_emo_add_last_pretrained16'
# model='emovdb_emo_add_pca_pretrained16'
# model='emovdb_emo_add_eis_word_pretrained16'
# model='emovdb_emo_add_eis_sentence_pretrained16'
# model='emovdb_emo_add_bert_cls_pretrained16'

# method='sem'
# model='emovdb_sem_mat_phone_pretrained16'
# model='emovdb_sem_mat_text_pretrained16'
# model='emovdb_sem_mat_bert_phone_pretrained16'
# model='emovdb_sem_mat_bert_text_pretrained16'

# choose checkpoint
# step='G_50000'
# step='G_100000'
step='G_150000'


# 1. Run `eval_1_make_kaldi_style_files.py` to rename the generated audio samples in the `source_model_test_wav` file corresponding to its transcript key. And generate related scp files. 
python3 vits/eval_datasets/eval_emovdb/eval_1_make_kaldi_style_files.py ${method} ${model} ${step}

# 2. Run `eval_2_unify_and_eval.sh` to downsample both model_wav and gt_wav to ensure they have the the same sampling rate, then eval.
# . /data/vitsGPT/vits/eval_datasets/eval_emovdb/eval_2_unify_and_eval_init.sh ${method} ${model} ${step} # You only need to run this once. But you can run below scripts many times. 
. vits/eval_datasets/eval_emovdb/eval_2_unify_and_eval.sh ${method} ${model} ${step}

# 3. Run `eval_3_mos.py` to evaluate UTMOS using the SpeechMOS framework. 
CUDA_VISIBLE_DEVICES=0 python3 vits/eval_datasets/eval_emovdb/eval_3_mos.py ${method} ${model} ${step}

# 4. Find a random English recipe in ESPnet, and use its code to do objective evaluation. Specifically, run `eval.sh` to evaluate MCD, ASR, F0 using the ESPnet framework. (You can also run this step before the step 3.)
# CUDA_VISIBLE_DEVICES=0 . espnet/egs2/libritts/tts1/eval.sh ${method} ${model} ${step} 
# option: run in the background
CUDA_VISIBLE_DEVICES=0 nohup espnet/egs2/libritts/tts1/eval.sh ${method} ${model} ${step} > eval.log 2>&1 & 