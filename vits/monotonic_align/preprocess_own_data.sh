# Cython-version Monotonoic Alignment Search
cd vits/monotonic_align
python vits/monotonic_align/setup.py build_ext --inplace

# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for LJSpeech and EmoV_DB have been already provided.
# python vits/preprocess.py --text_index 1 --filelists vits/filelists/ljs_audio_text_train_filelist.txt vits/filelists/ljs_audio_text_val_filelist.txt vits/filelists/ljs_audio_text_test_filelist.txt 
# python vits/preprocess.py --text_index 2 --filelists vits/filelists/vctk_audio_sid_text_train_filelist.txt vits/filelists/vctk_audio_sid_text_val_filelist.txt vits/filelists/vctk_audio_sid_text_test_filelist.txt
# python vits/preprocess.py --text_index 1 --filelists vits/filelists/jsut_audio_text_train_filelist.txt vits/filelists/jsut_audio_text_val_filelist.txt vits/filelists/jsut_audio_text_test_filelist.txt 
# python vits/preprocess.py --text_index 1 --filelists vits/filelists/onehour_ljs_audio_text_train_filelist.txt vits/filelists/tenmin_ljs_audio_text_train_filelist.txt
# python vits/ori_vits/preprocess.py --text_index 1 --filelists vits/filelists/emovdb_audio_text_train_filelist.txt vits/filelists/emovdb_audio_text_dev_filelist.txt vits/filelists/emovdb_audio_text_test_filelist.txt
python vits/ori_vits/preprocess.py --text_index 1 --filelists vits/filelists/librif_audio_text_train_filelist.txt vits/filelists/librif_audio_text_dev_filelist.txt vits/filelists/librif_audio_text_test_filelist.txt vits/filelists/librif_audio_text_all_filelist.txt
