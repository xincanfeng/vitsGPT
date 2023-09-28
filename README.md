# Text embedding with Language Models for TTS Synthesis

*Llama 2 + VITS*

### Paper Directions

1. Efficient semantic text embedding for TTS: capturing speech variations influenced by text
2. Text embedding for TTS: a survey
3. Text-edited TTS

* methods
  * add intelligent embedding. also, adding human-edit function is also useful. so that human can control the TTS using their label instead of the LM's label.
  * editing the text directly, e.g., remove the full stop or not.
  * curriculum learning

### Abstract

タイトル：（シンプルな高効果の）テキスト微細変動感知音声合成手法

研究背景：

近年の音声合成技術において、多様なスタイルの音声生成は可能となってきたが、テキストの微細な変更に対する感受性が不足している。特に、広告テキスト生成などの特定のアプリケーションにおいては、テキストの微小な変動が極めて重要である。従来のTTSモデルは、テキストの特徴を音声合成の指標として的確に捉えることが難しく、また、比較用のテキスト音声データの取得も困難である。この背景を踏まえ、事前学習済みモデルからのテキスト特徴抽出を活用した効率的なテキスト微細変動感知音声合成モデルの提案を目指す。

現在の対比学習の分野では、文の埋め込みに関するいくつかの研究や評価手法がありますが、異なる対比方法の分析やそれらがさまざまなモデルに及ぼす影響、さらにはモデル内部の注意力構造への影響に関する研究が行われています。しかし、音声合成への応用を評価基準として利用する研究はまだ完全には成熟していません。また、テキスト埋め込みが音声合成の応用にも使用されていますが、その方法は比較的単純で、基本的にはまずラベルを生成し、その後一定の微調整を行うというものです。この研究ギャップを埋めるために、我々は多くのテキスト埋め込み手法が音声合成の効果にどう影響するかを実験し、評価する研究を行う予定です。これにより、テキスト埋め込みを音声合成分野でより直接的かつ豊かに利用することが期待されます。

提案手法：

本研究では、事前学習済みの言語モデル（例：GPT、BERT）や知識グラフに基づくテキスト埋め込みモデルから、テキスト（或はその特徴）を抽出する。正サンプルに対して、TextAttackやNLPAugなどのツールを利用して、句読点の変更、単語の置換、文法の調整、感情の変換、逆翻訳、情報の曖昧化・具体化等の微細な修正を施し、対照的な負サンプルを生成する。その上で、**正サンプルと負サンプル間のコントラスト損失（或はそれらが生成されたpitch、energy、durationのコントラスト損失）を計算するによって、テキスト埋め込み（或はテキスト埋め込み情報を含むピッチ、エネルギー、持続時間の予測）の微細変動を感知できる音声合成を学習させる**。

実験設計：

提案モデルのシンプルさと効果性を追求するため、以下の設定でのテキスト埋め込み手法の効果性を検証分析する。

* Ablation Study
  * **どのテキスト関連トークンを活用するか (last, ave, pca, eis sentence, eis word)**
  * **テキスト埋め込み情報はどうやって音素情報と連携するのか**
  * * **add/attention to the text directly**
    * **add/attention to pitch, energy, and duration module**
  * さまざまなテキスト埋め込み方法、損失関数（例：Triplet Margin Loss、Contrastive Loss）の効果を実験的に比較評価する。
  * さらに、時間とリソースを許す限り、他のTTSモデルでの汎用性実験を行う。
* **Analysis**
  * **text and phoneme alighment**
  * **scale influence**
    * **number of tokens**
    * **number of dimensions**
  * **layer influence in the structure**

応用例：

提案モデルは、テキスト内容のニュアンスを重視する音声生成アプリケーションに適用可能であり、例としてオーディオブック、公式アナウンス、感情駆動型の音声合成などが挙げられる。

### Background and Proposal

Current research is more about attitude than emotion. Different from discrete and one-sidedly expressed attitude, emotion is usually continuous and harmoniously expressed by

* [emb_gt] word usage --> we want to learn it from the LM-generated vector embedding on the text.
* [emb_hm] pronunciation and tone (soft and gentle, calm and serious, impatient and unconcerned...) --> we can learn it from the human-annotated sentiment labels on the wav.
  * use Reparameterization Trick in the VAE

### Related Research

2023

ChatGPT-EDSS: Empathetic Dialogue Speech Synthesis Trained from ChatGPT-derived Context Word Embeddings [[paper](https://arxiv.org/abs/2305.13724)]

2022

Comparison and Combination of Sentence Embeddings Derived from Different Supervision Signals [[paper](https://arxiv.org/abs/2202.02990)]

2021

PnG BERT: Augmented BERT on Phonemes and Graphemes for Neural TTS [[paper](https://arxiv.org/abs/2103.15060)]

2020

Conversational End-to-End TTS for Voice Agent [[paper](https://arxiv.org/abs/2005.10438)]

2019

Neural TTS Stylization with Adversarial and Collaborative Games [[paper](https://openreview.net/pdf?id=ByzcS3AcYX)]

2018

Emo2Vec: Learning Generalized Emotion Representation by Multi-task Training [[paper](https://arxiv.org/abs/1809.04505)]

### Analysis

* BERTViz: Visualize Attention in NLP Models
* speaker classification analysis
  * different speakers have different variation when text semantic changes

## Methods

@training

text -> Llama -> emotional speech

@inference

text -> Llama -> emotional speech

### Model Structure

* Llama 2

  * Prompt tuning
  * Fine-tuning (if necessary)
    * Use model generated labels
    * Create human annotated labels
      * If the dataset can be used to evaluate more models fairly, then a new benchmark is created
  * Reinforcement Learning with Human Feedback (RLHF)
* VITS

  * @training
    * sentiment embedding from original text
      * add linear emotion embedding layer
      * embedding size is decided by hyper-parameter
    * sentiment embedding from human-annotated label text
      * embedding size is influenced by number&levels of sentiments in the label
    * sentiment embedding from LM-generated label text
      * embedding size is influenced by number&levels of sentiments in the label
  * @inference
    * text -> text sentiment weight + prontone sentiment weight -> speech

## Download models, weights, and datasets

1. Llama 2
   1. [model repository](https://github.com/facebookresearch/llama)
   2. [pretrained weight](https://ai.meta.com/resources/models-and-libraries/llama/)
2. VITS
   1. [model repository](https://github.com/jaywalnut310/vits/tree/main)
   2. [model weight](https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2)
3. datasets
   1. English
      1. [LJSpeech-1.1](https://keithito.com/LJ-Speech-Dataset/)
      2. [VCTK-Corpus-0.92](https://datashare.ed.ac.uk/handle/10283/3443)
      3. [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm)
      4. [LibriTTS](https://www.openslr.org/60/)
   2. Japanese
      1. [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)
      2. [J-MAC](https://sites.google.com/site/shinnosuketakamichi/research-topics/j-mac_corpus)
   3. Chinese
      1. [THCHS-30](http://www.openslr.org/18/)
