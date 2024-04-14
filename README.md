# **[Llama-VITS](https://arxiv.org/abs/2404.06714)**

This repository is the PyTorch implementation of Llama-VITS for enhanced TTS synthesis with semantic awareness extracted from a large-scale language model. 

## 1. Implemented Features:  
**Model with Checkpoints:** 
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

**Audio Demos**

## 2. Quick Generation
### 2.1 Configure the Checkpoint File
### 2.2 Generate Speech based on Trained Checkpoints



## 3. Train

### 3.1 Extracting Semantic Embeddings
#### 3.1.1 Extracting Semantic Embeddings From Llama
Download Llama2 weights and tokenizer
In order to download the Llama2 weights and tokenizer, please visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and accept their License.

#### 3.1.2 Extracting Semantic Embeddings From various BERT models

### 3.2 Training
#### 3.2.1 Train VITS with no Semantic Tokens  
#### 3.2.2 Train VITS with Global Semantic Tokens   
#### 3.2.3 Train VITS with Sequential Semantic Tokens  


## 4. Evaluation
### 4.1 Data Pre-processing



## **Citation**

If our work is useful to you, please cite our paper: "**Llama-VITS: Enhancing TTS Synthesis with Semantic Awareness**". [paper](https://arxiv.org/abs/2404.06714)
```
@misc{feng2024llamavits,
      title={Llama-VITS: Enhancing TTS Synthesis with Semantic Awareness}, 
      author={Xincan Feng and Akifumi Yoshimoto},
      year={2024},
      eprint={2404.06714},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
