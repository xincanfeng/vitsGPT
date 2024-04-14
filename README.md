# **[Llama-VITS](https://arxiv.org/abs/2404.06714)**

In our recent [paper](https://arxiv.org/abs/2404.06714), we propose Llama-VITS for enhanced TTS synthesis with semantic awareness extracted from a large-scale language model.  
This repository is the PyTorch implementation of Llama-VITS. Please visit our [demo](a github.io page to appear) for audio samples. 

## Implemented Features:  
**Model with Weights:** 
- [x] Llama-VITS  
- [x] BERT-VITS  
- [x] ORI-VITS  

**Evaluation Metrics:**
- [x] ESMOS  
- [x] UTMOS  
- [x] MCD  
- [x] CER  
- [x] WER  

**Datasets:**  
- [x] full LJSpeech  
- [x] 1-hour LJSpeech  
- [x] EmoV_DB_bea_sem  

## Pre-requisites 
0. Python >= 3.6
0. Clone this repository
0. Install python requirements. Please refer to [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak`
0. Download datasets
    1. Download and extract the LJSpeech dataset from [here](https://keithito.com/LJ-Speech-Dataset/), then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs vits/DUMMY1`
    1. Download and extract the 1-hour LJSpeech dataset from [here](a google drive to appear), then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs vits/DUMMY2`
    1. Download and extract the EmoV_DB_bea_sem dataset from [here](a google drive to appear), then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs vits/DUMMY3`
0. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.  
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace

# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for LJSpeech and EmoV_DB_bea_sem have been already provided.
# python preprocess.py --text_index 1 --filelists filelists/ljs_audio_text_train_filelist.txt filelists/ljs_audio_text_val_filelist.txt filelists/ljs_audio_text_test_filelist.txt 
# python preprocess.py --text_index 2 --filelists filelists/vctk_audio_sid_text_train_filelist.txt filelists/vctk_audio_sid_text_val_filelist.txt filelists/vctk_audio_sid_text_test_filelist.txt
```
Please refer to [preprocess_own_data.sh](vits/ori_vits/monotonic_align/preprocess_own_data.sh) for more configurations. 



## Extracting Semantic Embeddings
We provide code to extract semantic embeddings from Llama or various BERT models. 

Note that we also provide the [extracted semantic embeddings](a google drive page to apper). 

### Extracting Semantic Embeddings From Llama
0. Download Llama weights and tokenizer  
Use the Llama implementation in our repository which includes codes to extract the semantic embeddings in the final hidden layer. Then download the Llama weights and tokenizer from [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and accept their License. 

Once your request is approved, you will receive a signed URL over email. Then run the [download.sh](llama/download.sh) script, passing the URL provided when prompted to start the download.

Pre-requisites: Make sure you have `wget` and `md5sum` installed. Then run the script: `./download.sh`.

Keep in mind that the links expire after 24 hours and a certain amount of downloads. If you start seeing errors such as 403: Forbidden, you can always re-request a link.

Please refer to [Llama](https://github.com/meta-llama/llama/tree/main) repository if there are further related questions. 

### Extracting Semantic Embeddings From various BERT models




## Training
You can train the VITS model w/ or w/o semantic tokens using the scripts below. 

Note that we also provide the [pretrained models](a google drive page to appear).

### Training VITS with no semantic tokens  
```sh
python vits/ori_vits/train.py -c vits/configs/ljs_base.json -m ljs_base
```
Please refer to [train.sh](vits/ori_vits/train.sh) for specific configurations of different datasets.
### Training VITS with global semantic tokens   
```sh
python vits/emo_vits/emo_train.py -c vits/configs/ljs_sem_ave.json -m ljs_emo_add_ave
```
Please refer to [emo_train.sh](vits/emo_vits/emo_train.sh) for specific configurations of different datasets and global tokens.
### Training VITS with sequential semantic tokens  
```sh
python vits/sem_vits/sem_train.py -c vits/configs/ljs_sem_mat_text.json -m ljs_sem_mat_text
```
Please refer to [sem_train.sh](vits/sem_vits/sem_train.sh) for specific configurations of different datasets and sequential tokens. ("mat" in the sequential tokens' file name means "matrix", because compared to global token which is mathematically represented by a single vector, sequential token is represented by a matrix for each sentence transcript.)


## Inferencing
See [inference.ipynb](vits/ori_vits/inference.ipynb) as an easy example to understand how to inference on any text.  

Configure the model weights w/ or w/o extracted semantic tokens in the files below, then you can inference on test data transcripts. 
Use [infer_test.ipynb](vits/ori_vits/infer_test.ipynb) for inferencing with no semantic tokens on test data transcripts.  
Use [emo_infer_test.ipynb](vits/emo_vits/emo_infer_test.ipynb) for inferencing with global semantic tokens on test data transcripts.  
Use [sem_infer_test.ipynb](vits/sem_vits/sem_infer_test.ipynb) for inferencing with sequential semantic tokens on test data transcripts. 


## Evaluation
### Data Pre-processing



## **Citation**
If our work is useful to you, please cite our paper: "**Llama-VITS: Enhancing TTS Synthesis with Semantic Awareness**". [paper](https://arxiv.org/abs/2404.06714)
```sh
@misc{feng2024llamavits,
      title={Llama-VITS: Enhancing TTS Synthesis with Semantic Awareness}, 
      author={Xincan Feng and Akifumi Yoshimoto},
      year={2024},
      eprint={2404.06714},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
