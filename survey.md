# Text Representation for Enhancing Emotional Richness in Speech Synthesis

## Learning Methods

Contrastive Learning Models for Sentence Representations [[paper](https://dl.acm.org/doi/pdf/10.1145/3593590)]

Investigating Views for Contrastive Learning of Language Representations [[paper](https://web.stanford.edu/class/cs224n/reports/custom_116741022.pdf)]

## LM+TTS

2022

PromptTTS: Controllable Text-to-Speech with Text Descriptions [[paper](https://arxiv.org/abs/2211.12171)]

2023

NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers [[paper](https://arxiv.org/abs/2304.09116)]

ChatGPT-EDSS: Empathetic Dialogue Speech Synthesis Trained from ChatGPT-derived Context Word Embeddings [[paper](https://arxiv.org/abs/2305.13724)]

PromptTTS: controllable text-to-speech with text descriptions [[paper](https://speechresearch.github.io/prompttts/#Part%203)]

Natural Language Supervision for General-Purpose Audio Representations [[paper](https://arxiv.org/abs/2309.05767)]

## LM

### Models

2023

(**Llama 2**) Llama 2: Open Foundation and Fine-Tuned Chat Models [[paper](https://arxiv.org/abs/2307.09288)] [[source](https://ai.meta.com/resources/models-and-libraries/llama/)] [[source code](https://github.com/facebookresearch/llama)]

### Emotionalizing LM

2019

Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset [[paper](https://aclanthology.org/P19-1534/)]

### Tutorials

[Hugging face NLP Course](https://huggingface.co/learn/nlp-course/zh-CN/chapter6/2?fw=pt)

## KG

2022

Knowledge Graph Augmented Network Towards Multiview Representation Learning for Aspect-based Sentiment Analysis [[paper](https://arxiv.org/abs/2201.04831)]

Knowledge-Interactive Network with Sentiment Polarity Intensity-Aware Multi-Task Learning for Emotion Recognition in Conversations [[paper](https://aclanthology.org/2021.findings-emnlp.245.pdf)]

KESA: A Knowledge Enhanced Approach To Sentiment Analysis [[paper](https://aclanthology.org/2022.aacl-main.58.pdf)]

## TTS

### Models

2017

(**Tacotron**) Tacotron: Towards End-to-End Speech Synthesis [[paper](https://arxiv.org/abs/1703.10135)] [source code] [[official demo](https://google.github.io/tacotron/)]

2020

(**FastPitch**) FastPitch: Parallel Text-to-speech with Pitch Prediction [[paper](https://arxiv.org/abs/2006.06873)] [[source code](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch)] [[official demo](https://fastpitch.github.io/)] [[customize demo](https://huggingface.co/nvidia/tts_en_fastpitch)]

(**FastSpeech2**) FastSpeech 2: Fast and High-Quality End-to-End Text to Speech [[paper](https://arxiv.org/abs/2006.04558)] [source code] [[official demo](https://speechresearch.github.io/fastspeech2/)] [[customize demo](https://huggingface.co/facebook/fastspeech2-en-ljspeech)]

2021

(**VITS**) Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech [[paper](https://arxiv.org/abs/2106.06103)] [[source code](https://github.com/jaywalnut310/vits)] [[official demo](https://jaywalnut310.github.io/vits-demo/index.html)]

2022

(**JETS**) JETS: Jointly Training FastSpeech2 and HiFi-GAN for End to End Text to Speech [[paper](https://arxiv.org/abs/2203.16852)] [source code] [[official demo](https://imdanboy.github.io/interspeech2022/)]

(**AudioLM**) AudioLM: a Language Modeling Approach to Audio Generation [[paper]()] [source code] [[offcial demo](https://google-research.github.io/seanet/audiolm/examples/)]

2023

(**VALL-E**) Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers [[paper](https://arxiv.org/abs/2301.02111)] [source code] [[official demo](https://www.microsoft.com/en-us/research/project/vall-e-x/)]

(**SPEAR-TTS**) Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal Supervision [[paper](https://arxiv.org/abs/2302.03540)] [source code] [[official demo](https://google-research.github.io/seanet/speartts/examples/)]

(**Voicebox**) Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale [[paper](https://scontent-nrt1-1.xx.fbcdn.net/v/t39.8562-6/354636794_599417672291955_3799385851435258804_n.pdf?_nc_cat=101&ccb=1-7&_nc_sid=ad8a9d&_nc_ohc=YQ53jZfBSoMAX_-Vc2H&_nc_ht=scontent-nrt1-1.xx&oh=00_AfBI7Eq7xwLmSEDKp1wDCGVwTuNlyCALXpy_3j1ZWAM8Dg&oe=649C64F1)] [source code] [[official demo](https://voicebox.metademolab.com/)]

(**AudioPaLM**) AudioPaLM: A Large Language Model That Can Speak and Listen [[paper](https://arxiv.org/abs/2306.12925)] [source code] [[official demo](https://google-research.github.io/seanet/audiopalm/examples/)]

(**VITS2**) VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design [[paper](https://arxiv.org/abs/2307.16430)] [source code] [[official demo](https://vits-2.github.io/demo/)]

### Stylizing TTS

ニューラルボコーダー論文25本ノック [[artical](https://qiita.com/4wavetech/items/28441857d2139aecaf6a)]

Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis [[paper](https://arxiv.org/pdf/1803.09017.pdf)]

Neural TTS Stylization with Adversarial and Collaborative Games [[paper](https://openreview.net/pdf?id=ByzcS3AcYX)]

HiFiSinger: Towards High-Fidelity Neural Singing Voice Synthesis [[paper](https://arxiv.org/abs/2009.01776)] [[official demo](https://speechresearch.github.io/hifisinger/)]

Xiaoicesing 2: A High-Fidelity Singing Voice Synthesizer Based on Generative Adversarial Network [[paper](https://arxiv.org/abs/2210.14666)]

### Codebases

[coqui-ai](https://github.com/coqui-ai/TTS): Tacotron, FastPitch, FastSpeech, VITS, YourTTS, HiFiGAN

[audiolm-pytorch](https://github.com/lucidrains/audiolm-pytorch): AudioLM, VALL-E

[espnet](https://github.com/espnet/espnet): Tacotron, FastSpeech, VITS, JETS

## Metrics

[espnet evaluation]()

SpeechMOS [[official code](https://github.com/tarepan/SpeechMOS)]

Finetune SSL models for MOS prediction [[official code](https://github.com/nii-yamagishilab/mos-finetune-ssl)]

UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022 [[paper](https://arxiv.org/abs/2204.02152)] [[official code](https://github.com/sarulab-speech/UTMOS22)]

[VoiceMOS](https://voicemos-challenge-2023.github.io/)

SpeechLMScore: Evaluating speech generation using speech language model [[paper](https://arxiv.org/abs/2212.04559)]

CROWDMOS: An approach for crowdsourcing mean opinion score studies [[paper](https://ieeexplore.ieee.org/document/5946971)]

Significant test: mean-opinion-score 0.0.2 [[code](https://pypi.org/project/mean-opinion-score/)]

### Datasets

LJ Speech [[download](https://keithito.com/LJ-Speech-Dataset/)] [[example code](https://github.com/keithito/tacotron)]

JSUT [[example config](https://github.com/espnet/espnet/blob/master/egs2/jsut/tts1/conf/tuning/train_vits.yaml)]

[SER](https://superkogito.github.io/SER-datasets/)

[ESD](https://hltsingapore.github.io/ESD/index.html)

[ita-corpus](https://github.com/mmorise/ita-corpus/blob/main/recitation_transcript_utf8.txt)

### Tutorials

[応用音響学](https://www.sp.ipc.i.u-tokyo.ac.jp/~saruwatari/AA2018_02.pdf)

[AI音声合成䛾技術動向](https://drive.google.com/file/d/1w8LtI9Sz31Qb4AtdWWmuy2AYa4OZ4fNh/view)

## Journals and Conferences

[Journal of The Acoustical Society of America (JASA)](https://pubs.aip.org/asa/jel)

[IEEE Signal Processing Letters (SPL)](https://signalprocessingsociety.org/publications-resources/ieee-signal-processing-letters)

[日本音響学会](https://acoustics.jp/)

## Coding Tricks

[Profiler Recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

## Related Work

- PnG BERT: Augmented BERT on Phonemes and Graphemes for Neural TTS [[paper](https://arxiv.org/pdf/2103.15060.pdf)]
- Improving Prosody with Linguistic and Bert Derived Features in Multi-Speaker Based Mandarin Chinese Neural TTS [[paper](https://ieeexplore.ieee.org/abstract/document/9054337)]
- Mixed-Phoneme BERT: Improving BERT with Mixed Phoneme and Sup-Phoneme Representations for Text to Speech [[paper](https://arxiv.org/pdf/2203.17190.pdf)]
- Unified Mandarin TTS Front-end Based on Distilled BERT Model [[paper](https://arxiv.org/pdf/2012.15404.pdf)]
- BERT, can HE predict contrastive focus? Predicting and controlling prominence in neural TTS using a language model [[paper](https://arxiv.org/abs/2207.01718)]
- Investigation of Japanese PnG BERT Language Model in Text-to-Speech Synthesis for Pitch Accent Language [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9829304)]
- Improving the Prosody of RNN-based English Text-To-Speech Synthesis by Incorporating a BERT model [[paper](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/09d96197b11583edbc2349c29a9f0cf7777f4def.pdf)]
- A Universal Bert-Based Front-End Model for Mandarin Text-To-Speech Synthesis [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9414935)]
- Phoneme-Level Bert for Enhanced Prosody of Text-To-Speech with Grapheme Predictions [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10097074)] --- no access
- Text aware Emotional Text-to-speech with BERT [[paper](https://www.researchgate.net/profile/Shubham-Bansal-22/publication/363646884_Text_aware_Emotional_Text-to-speech_with_BERT/links/6337bc819cb4fe44f3f022a2/Text-aware-Emotional-Text-to-speech-with-BERT.pdf)]
- Polyphone Disambiguation and Accent Prediction Using Pre-Trained Language Models in Japanese TTS Front-End [[paper](https://ieeexplore.ieee.org/iel7/9745891/9746004/09746212.pdf)] --- no access
- Mixer-TTS: Non-Autoregressive, Fast and Compact Text-to-Speech Model Conditioned on Language Model Embeddings [[paper](https://ieeexplore.ieee.org/iel7/9745891/9746004/09746107.pdf)] --- no access
- Expressive, Variable, and Controllable Duration Modelling in TTS [[paper](https://arxiv.org/pdf/2206.14165)]
- EE-TTS: Emphatic Expressive TTS with Linguistic Information [[paper](https://arxiv.org/pdf/2305.12107)]
- Mixed Orthographic/Phonemic Language Modeling: Beyond Orthographically Restricted Transformers (BORT) [[paper](https://aclanthology.org/2023.repl4nlp-1.18.pdf)]
- Whispering LLaMA: A Cross-Modal Generative Error Correction Framework for Speech Recognition [[paper](https://arxiv.org/pdf/2310.06434.pdf)]
- Vec-Tok Speech: speech vectorization and tokenization for neural speech generation [[paper](https://arxiv.org/abs/2310.07246)]
