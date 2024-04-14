# LJSpeech
# add
CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/ljs_sem_ave.json -m ljs_emo_add_ave
CUDA_VISIBLE_DEVICES=1 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/ljs_sem_last.json -m ljs_emo_add_last
CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/ljs_sem_pca.json -m ljs_emo_add_pca
CUDA_VISIBLE_DEVICES=1 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/ljs_sem_eis_word.json -m ljs_emo_add_eis_word
CUDA_VISIBLE_DEVICES=2 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/ljs_sem_eis_sentence.json -m ljs_emo_add_eis_sentence
CUDA_VISIBLE_DEVICES=1 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/ljs_bert_cls.json -m ljs_emo_add_bert_cls

CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/onehour_ljs_sem_ave.json -m onehour_ljs_emo_add_ave
CUDA_VISIBLE_DEVICES=1 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/onehour_ljs_sem_last.json -m onehour_ljs_emo_add_last
CUDA_VISIBLE_DEVICES=2 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/onehour_ljs_sem_pca.json -m onehour_ljs_emo_add_pca
CUDA_VISIBLE_DEVICES=1 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/onehour_ljs_sem_eis_word.json -m onehour_ljs_emo_add_eis_word
CUDA_VISIBLE_DEVICES=1 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/onehour_ljs_sem_eis_sentence.json -m onehour_ljs_emo_add_eis_sentence
CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/onehour_ljs_bert_cls.json -m onehour_ljs_emo_add_bert_cls

# CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/tenmin_ljs_sem_ave.json -m tenmin_ljs_emo_add_ave
# CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/tenmin_ljs_sem_last.json -m tenmin_ljs_emo_add_last
# CUDA_VISIBLE_DEVICES=2 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/tenmin_ljs_sem_pca.json -m tenmin_ljs_emo_add_pca
# CUDA_VISIBLE_DEVICES=1 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/tenmin_ljs_sem_eis_word.json -m tenmin_ljs_emo_add_eis_word
# CUDA_VISIBLE_DEVICES=1 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/tenmin_ljs_sem_eis_sentence.json -m tenmin_ljs_emo_add_eis_sentence

# att
# CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/ljs_sem_ave.json -m ljs_emo_att_ave
# CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/ljs_sem_last.json -m ljs_emo_att_last 
# CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/ljs_sem_pca.json -m ljs_emo_att_pca 
# CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/ljs_sem_eis_word.json -m ljs_emo_att_eis_word 
# CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/ljs_sem_eis_sentence.json -m ljs_emo_att_eis_sentence


# TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=1 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/tenmin_ljs_sem_ave.json -m test






# EmoV_DB
# add
CUDA_VISIBLE_DEVICES=1 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/emovdb_sem_ave.json -m emovdb_emo_add_ave
CUDA_VISIBLE_DEVICES=2 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/emovdb_sem_last.json -m emovdb_emo_add_last
CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/emovdb_sem_pca.json -m emovdb_emo_add_pca
CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/emovdb_sem_eis_word.json -m emovdb_emo_add_eis_word
CUDA_VISIBLE_DEVICES=1 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/emovdb_sem_eis_sentence.json -m emovdb_emo_add_eis_sentence
CUDA_VISIBLE_DEVICES=2 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/emovdb_bert_cls.json -m emovdb_emo_add_bert_cls


CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/emovdb_sem_ave16.json -m emovdb_emo_add_ave_pretrained16
CUDA_VISIBLE_DEVICES=1 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/emovdb_sem_last16.json -m emovdb_emo_add_last_pretrained16
CUDA_VISIBLE_DEVICES=2 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/emovdb_sem_pca16.json -m emovdb_emo_add_pca_pretrained16
CUDA_VISIBLE_DEVICES=1 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/emovdb_sem_eis_word16.json -m emovdb_emo_add_eis_word_pretrained16
CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/emovdb_sem_eis_sentence16.json -m emovdb_emo_add_eis_sentence_pretrained16
CUDA_VISIBLE_DEVICES=1 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/emovdb_bert_cls16.json -m emovdb_emo_add_bert_cls_pretrained16

# LibriTTS_filtered
# add
CUDA_VISIBLE_DEVICES=1 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/librif_sem_ave.json -m librif_emo_add_ave
# CUDA_VISIBLE_DEVICES=2 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/librif_sem_last.json -m librif_emo_add_last
# CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/librif_sem_pca.json -m librif_emo_add_pca
# CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/librif_sem_eis_word.json -m librif_emo_add_eis_word
# CUDA_VISIBLE_DEVICES=1 python /data/vitsGPT/vits/emo_vits/emo_train.py -c /data/vitsGPT/vits/configs/librif_sem_eis_sentence.json -m librif_emo_add_eis_sentence