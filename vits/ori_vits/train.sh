# LJSpeech
CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/ori_vits/train.py -c /data/vitsGPT/vits/configs/ljs_base.json -m ljs_base
CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/ori_vits/train.py -c /data/vitsGPT/vits/configs/onehour_ljs_sem_mat_text.json -m onehour_ljs_base
# CUDA_VISIBLE_DEVICES=2 python /data/vitsGPT/vits/ori_vits/train.py -c /data/vitsGPT/vits/configs/tenmin_ljs_sem_mat_text.json -m tenmin_ljs_base

# EmoV_DB_bea_filtered
CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/ori_vits/train.py -c /data/vitsGPT/vits/configs/emovdb_base.json -m emovdb_base
CUDA_VISIBLE_DEVICES=1 python /data/vitsGPT/vits/ori_vits/train.py -c /data/vitsGPT/vits/configs/emovdb_base16.json -m emovdb_base_pretrained16

# LibriTTS_filtered
CUDA_VISIBLE_DEVICES=2 python /data/vitsGPT/vits/ori_vits/train.py -c /data/vitsGPT/vits/configs/librif_base.json -m librif_base
# CUDA_VISIBLE_DEVICES=0 python /data/vitsGPT/vits/ori_vits/train.py -c /data/vitsGPT/vits/configs/librif_base.json -m librif_base_pretrained