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

## Quick Inference 
We also provide the [pretrained models](a google drive page to appear). Follow the steps below to do quick inference using the pretrained models. 

0. Python >= 3.6
0. Clone this repository
0. Install python requirements. Please refer to [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak`
0. Download datasets
    1. Download and extract the LJSpeech dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1`
    1. Download and extract the 1-hour LJSpeech dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY2`
    1. Download and extract the EmoV_DB_bea_sem dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY3`
0. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.  
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace

# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for LJSpeech and EmoV_DB_bea_sem have been already provided.
# python preprocess.py --text_index 1 --filelists filelists/ljs_audio_text_train_filelist.txt filelists/ljs_audio_text_val_filelist.txt filelists/ljs_audio_text_test_filelist.txt 
# python preprocess.py --text_index 2 --filelists filelists/vctk_audio_sid_text_train_filelist.txt filelists/vctk_audio_sid_text_val_filelist.txt filelists/vctk_audio_sid_text_test_filelist.txt
```
Please refer to [preprocess_own_data.sh](preprocess_own_data.sh) for more configurations. 






## Extracting Semantic Embeddings
### Extracting Semantic Embeddings From Llama

1. Download Llama weights and tokenizer  
For Llama-related questions, please refer to [Llama](https://github.com/meta-llama/llama/tree/main) repository. Basicly, in order to download the Llama weights and tokenizer, one need to visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and accept their License. 

2. 

### Extracting Semantic Embeddings From various BERT models




## Training

Please refer to this document for specific configurations of different models and datasets.

### Training VITS with no semantic tokens  
```sh
python vitsGPT/vits/ori_vits/train.py -c /vitsGPT/vits/configs/ljs_base.json -m ljs_base
```
### Training VITS with global semantic tokens   
### Training VITS with sequential semantic tokens  


## Inferencing
See [inference.ipynb](inference.ipynb) for an easy example.  
See [infer_test.ipynb](infer_test.ipynb) for inferencing using VITS with no semantic tokens on test data transcripts.  
See [emo_infer_test.ipynb](emo_infer_test.ipynb) for inferencing using VITS with global semantic tokens on test data transcripts.  
See [sem_infer_test.ipynb](sem_infer_test.ipynb) for inferencing using VITS with sequential semantic tokens on test data transcripts. 


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
