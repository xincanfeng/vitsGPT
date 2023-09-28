# Analyzing Text Sentiment with Language Models to Enhance Emotional Richness in Speech Synthesis

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

[espnet evaluation]()

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
