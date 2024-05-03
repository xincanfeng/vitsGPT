# LJSpeech
# batchsize 64
CUDA_VISIBLE_DEVICES=0 python sem_train.py -c configs/onehour_ljs_sem_mat_text.json -m onehour_ljs_sem_mat_text
CUDA_VISIBLE_DEVICES=1 python sem_train.py -c configs/onehour_ljs_sem_mat_phone.json -m onehour_ljs_sem_mat_phone
CUDA_VISIBLE_DEVICES=0 python sem_train.py -c configs/onehour_ljs_bert_text.json -m onehour_ljs_sem_mat_bert_text
CUDA_VISIBLE_DEVICES=0 python sem_train.py -c configs/onehour_ljs_bert_phone.json -m onehour_ljs_sem_mat_bert_phone

CUDA_VISIBLE_DEVICES=0 python sem_train.py -c configs/ljs_sem_mat_text.json -m ljs_sem_mat_text
CUDA_VISIBLE_DEVICES=0 python sem_train.py -c configs/ljs_sem_mat_phone.json -m ljs_sem_mat_phone
CUDA_VISIBLE_DEVICES=0 python sem_train.py -c configs/ljs_bert_text.json -m ljs_sem_mat_bert_text
CUDA_VISIBLE_DEVICES=0 python sem_train.py -c configs/ljs_bert_phone.json -m ljs_sem_mat_bert_phone


# EmoV_DB
# batchsize 16
CUDA_VISIBLE_DEVICES=0 python sem_train.py -c configs/emovdb_sem_mat_text16.json -m emovdb_sem_mat_text_pretrained16
CUDA_VISIBLE_DEVICES=1 python sem_train.py -c configs/emovdb_sem_mat_phone16.json -m emovdb_sem_mat_phone_pretrained16
CUDA_VISIBLE_DEVICES=0 python sem_train.py -c configs/emovdb_bert_text16.json -m emovdb_sem_mat_bert_text_pretrained16
CUDA_VISIBLE_DEVICES=1 python sem_train.py -c configs/emovdb_bert_phone16.json -m emovdb_sem_mat_bert_phone_pretrained16


