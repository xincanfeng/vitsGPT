# LJSpeech
# batchsize 64
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/ljs_base.json -m ljs_base
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/onehour_ljs_sem_mat_text.json -m onehour_ljs_base


# EmoV_DB_bea_sem
# batchsize 64
# CUDA_VISIBLE_DEVICES=0 python train.py -c configs/emovdb_base.json -m emovdb_base
# batchsize 16
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/emovdb_base16.json -m emovdb_base_pretrained16
