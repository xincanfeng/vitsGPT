# LJSpeech
# add
# batchsize 64
CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/ljs_sem_ave.json -m ljs_emo_add_ave 
CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/ljs_sem_last.json -m ljs_emo_add_last 
CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/ljs_sem_pca.json -m ljs_emo_add_pca 
CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/ljs_sem_eis_word.json -m ljs_emo_add_eis_word 
CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/ljs_sem_eis_sentence.json -m ljs_emo_add_eis_sentence 
CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/ljs_bert_cls.json -m ljs_emo_add_bert_cls 

CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/onehour_ljs_sem_ave.json -m onehour_ljs_emo_add_ave 
CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/onehour_ljs_sem_last.json -m onehour_ljs_emo_add_last 
CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/onehour_ljs_sem_pca.json -m onehour_ljs_emo_add_pca 
CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/onehour_ljs_sem_eis_word.json -m onehour_ljs_emo_add_eis_word 
CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/onehour_ljs_sem_eis_sentence.json -m onehour_ljs_emo_add_eis_sentence 
CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/onehour_ljs_bert_cls.json -m onehour_ljs_emo_add_bert_cls 


# EmoV_DB
# add 
# batchsize 64
# CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/emovdb_sem_ave.json -m emovdb_emo_add_ave 
# CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/emovdb_sem_last.json -m emovdb_emo_add_last 
# CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/emovdb_sem_pca.json -m emovdb_emo_add_pca 
# CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/emovdb_sem_eis_word.json -m emovdb_emo_add_eis_word 
# CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/emovdb_sem_eis_sentence.json -m emovdb_emo_add_eis_sentence 
# CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/emovdb_bert_cls.json -m emovdb_emo_add_bert_cls 

# batchsize 16
CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/emovdb_sem_ave16.json -m emovdb_emo_add_ave_pretrained16 
CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/emovdb_sem_last16.json -m emovdb_emo_add_last_pretrained16 
CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/emovdb_sem_pca16.json -m emovdb_emo_add_pca_pretrained16 
CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/emovdb_sem_eis_word16.json -m emovdb_emo_add_eis_word_pretrained16 
CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/emovdb_sem_eis_sentence16.json -m emovdb_emo_add_eis_sentence_pretrained16 
CUDA_VISIBLE_DEVICES=0 python emo_train.py -c configs/emovdb_bert_cls16.json -m emovdb_emo_add_bert_cls_pretrained16 
