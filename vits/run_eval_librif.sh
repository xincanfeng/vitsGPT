#!/bin/bash

# choose model
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

# choose checkpoint
step='G_50000'
# step='G_100000'
# step='G_150000'


# 1. Run `eval_1_make_kaldi_style_files.py` to rename the generated audio samples in the `source_model_test_wav` file corresponding to its transcript key. And generate related scp files. 
python3 vits/eval_datasets/eval_librif/eval_1_make_kaldi_style_files.py ${method} ${model} ${step}

# 2. Run `eval_2_unify_and_eval.sh` to downsample both model_wav and gt_wav to ensure they have the the same sampling rate, then eval.
. vits/eval_datasets/eval_librif/eval_2_unify_and_eval_init.sh ${method} ${model} ${step} # You only need to run this once. But you can run below scripts many times. 
# . vits/eval_librif/eval_2_unify_and_eval.sh ${method} ${model} ${step}

# 3. Find a random English recipe in ESPnet, and use its code to do objective evaluation. Specifically, run `eval.sh` to evaluate MCD, ASR, F0 using the ESPnet framework. (You can also run this step after the step 4.)
CUDA_VISIBLE_DEVICES=0 . espnet/egs2/libritts/tts1/eval.sh ${method} ${model} ${step} 
# option: run in the background
# CUDA_VISIBLE_DEVICES=0 nohup espnet/egs2/libritts/tts1/eval.sh ${method} ${model} ${step} > eval.log 2>&1 & 

# 4. Run `eval_3_mos.py` to evaluate UTMOS using the SpeechMOS framework. 
CUDA_VISIBLE_DEVICES=0 python3 vits/eval_datasets/eval_librif/eval_3_mos.py ${method} ${model} ${step}