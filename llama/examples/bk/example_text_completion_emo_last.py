# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import torch
from llama import Llama


output_file_name = 'ljs_audio_gt_last'

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
    input_file: str = 'datasets/LJSpeech-1.1/metadata_copy10.csv',
    output_file: str = f"vits/filelists/{output_file_name}_5120.pt",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 256,
    max_gen_len: int = 64,
    max_batch_size: int = 8,
):
    
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    audiopaths, prompts = load_sentences_from_file(input_file)

    output_dict = {}

    results = generator.text_completion(
        prompts,
        max_gen_len=max_seq_len,
        temperature=temperature,
        top_p=top_p,
    )

    # 紧接着generation，调用 get_promt_last_token_embedding 方法
    # h_last_real_token_b, h_ave_real_token_b, h_last_real_token_slt, h_ave_real_token_slt, h_pca_real_token_slt = generator.get_text_prompt_token_embedding()
    _, _, h_last_real_token_slt, _, _ = generator.get_text_prompt_token_embedding()
    # _, _, _, h_ave_real_token_slt, _ = generator.get_text_prompt_token_embedding()
    # _, _, _, _, h_pca_real_token_slt = generator.get_text_prompt_token_embedding()

    gt_embeddings = h_last_real_token_slt
    # gt_embeddings = h_ave_real_token_slt
    # gt_embeddings = h_pca_real_token_slt

    for audiopath, embedding in zip(audiopaths, gt_embeddings):
        output_dict[audiopath] = embedding.cpu()
        
    for audiopath, prompt, result, embedding in zip(audiopaths, prompts, results, gt_embeddings):
        print(f"geting embedding for {output_file_name}:")
        print(audiopath)
        print(prompt)
        print(result)
        print(embedding[:10])
        print("\n==================================\n")

    torch.save(output_dict, output_file)

if __name__ == "__main__":
    fire.Fire(main)