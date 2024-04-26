# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import fire
from llama import Llama
import torch

def load_sentences_from_file(input_file: str):
    """
    从给定的文件中加载句子。
    文件的格式应为: ID|句子1|句子2
    """
    audiopaths = []
    sentences = []
    with open(input_file, 'r') as file:
        for line in file:
            audiopath, sentence, _ = line.strip().split('|')
            audiopath = "DUMMY1/" + audiopath + ".wav" # 修改此行以适配正确的音频链接
            audiopaths.append(audiopath)
            sentences.append(sentence)
    return audiopaths, sentences

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    input_file: str = '/data/vitsGPT/datasets/LJSpeech-1.1/metadata_copy10.csv',
    output_file: str = '/data/vitsGPT/vits/filelists/ljs_audio_gt_eis_word_5120.pt',
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 16,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    audiopaths, sentences = load_sentences_from_file(input_file)

    output_dict = {}
    # 为每个句子构建对话
    for sentence in sentences:
        dialogs=[
            [
                {"role": "system", "content": "Always answer within a word."},
                {"role": "user", "content": f"what is the emotion of the sentence: {sentence}"},
            ],
            [
                {"role": "system", "content": "Always answer within a word."},
                {"role": "user", "content": f"what is the intention of the sentence: {sentence}"},
            ],
            [
                {"role": "system", "content": "Always answer within a word."},
                {"role": "user", "content": f"what is the speaking style of the sentence: {sentence}"},
            ],
        ]

        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        
        h_eis_ave_real_token_slt = generator.get_chat_prompt_token_embedding()

        gt_embeddings = h_eis_ave_real_token_slt

        for audiopath, embedding in zip(audiopaths, gt_embeddings):
            output_dict[audiopath] = embedding.cpu()

        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            print("\n==================================\n")

    # 保存字典为PyTorch的.pt文件
    torch.save(output_dict, output_file) # -1.0436, -0.8646

if __name__ == "__main__":
    fire.Fire(main)