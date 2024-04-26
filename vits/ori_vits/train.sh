# LJSpeech
# batchsize 64
CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/ori_vits/train.py -c /data/vitsGPT/vits/configs/ljs_base.json -m ljs_base
CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/ori_vits/train.py -c /data/vitsGPT/vits/configs/onehour_ljs_sem_mat_text.json -m onehour_ljs_base


# EmoV_DB_bea_sem
# batchsize 64
CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/ori_vits/train.py -c /data/vitsGPT/vits/configs/emovdb_base.json -m emovdb_base
# batchsize 16
CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/ori_vits/train.py -c /data/vitsGPT/vits/configs/emovdb_base16.json -m emovdb_base_pretrained16
