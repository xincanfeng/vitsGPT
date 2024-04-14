# Cython-version Monotonoic Alignment Search
cd /data/vitsGPT/vits/monotonic_align
python /data/vitsGPT/vits/monotonic_align/setup.py build_ext --inplace

# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for LJSpeech and EmoV_DB have been already provided.
# python /data/vitsGPT/vits/preprocess.py --text_index 1 --filelists /data/vitsGPT/vits/filelists/ljs_audio_text_train_filelist.txt /data/vitsGPT/vits/filelists/ljs_audio_text_val_filelist.txt /data/vitsGPT/vits/filelists/ljs_audio_text_test_filelist.txt 
# python /data/vitsGPT/vits/preprocess.py --text_index 2 --filelists /data/vitsGPT/vits/filelists/vctk_audio_sid_text_train_filelist.txt /data/vitsGPT/vits/filelists/vctk_audio_sid_text_val_filelist.txt /data/vitsGPT/vits/filelists/vctk_audio_sid_text_test_filelist.txt
# python /data/vitsGPT/vits/preprocess.py --text_index 1 --filelists /data/vitsGPT/vits/filelists/jsut_audio_text_train_filelist.txt /data/vitsGPT/vits/filelists/jsut_audio_text_val_filelist.txt /data/vitsGPT/vits/filelists/jsut_audio_text_test_filelist.txt 
# python /data/vitsGPT/vits/preprocess.py --text_index 1 --filelists /data/vitsGPT/vits/filelists/onehour_ljs_audio_text_train_filelist.txt /data/vitsGPT/vits/filelists/tenmin_ljs_audio_text_train_filelist.txt
# python /data/vitsGPT/vits/ori_vits/preprocess.py --text_index 1 --filelists /data/vitsGPT/vits/filelists/emovdb_audio_text_train_filelist.txt /data/vitsGPT/vits/filelists/emovdb_audio_text_dev_filelist.txt /data/vitsGPT/vits/filelists/emovdb_audio_text_test_filelist.txt
python /data/vitsGPT/vits/ori_vits/preprocess.py --text_index 1 --filelists /data/vitsGPT/vits/filelists/librif_audio_text_train_filelist.txt /data/vitsGPT/vits/filelists/librif_audio_text_dev_filelist.txt /data/vitsGPT/vits/filelists/librif_audio_text_test_filelist.txt /data/vitsGPT/vits/filelists/librif_audio_text_all_filelist.txt
