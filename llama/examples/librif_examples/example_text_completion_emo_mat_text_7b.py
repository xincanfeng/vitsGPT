# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import torch
from llama import Llama


output_file_name = 'librif_audio_sem_mat_text'

def load_sentences_from_file(input_file: str, batch_size: int):
    """
    从给定的文件中加载句子，并按batch_size返回批次。
    文件的格式应为: ID|句子1|句子2
    """
    audiopaths = []
    sentences = []
    batches = []
    
    with open(input_file, 'r') as file:
        for line in file:
            audiopath, sentence = line.strip().split('|')
            audiopaths.append(audiopath)
            sentences.append(sentence)
            
            if len(sentences) == batch_size:
                batches.append((audiopaths, sentences))
                audiopaths, sentences = [], []
                
    if audiopaths and sentences:  # handle the last batch if it's not empty
        batches.append((audiopaths, sentences))
        
    return batches

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    input_file: str = '/data/vitsGPT/vits/filelists/librif_audio_text_all_filelist.txt',
    output_file: str = f"/data/vitsGPT/vits/filelists/{output_file_name}_t4096.pt",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 256,
    max_gen_len: int = 64,
    max_batch_size: int = 50, # 本文件中每次最大输入的句子数
):
    
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    batches = load_sentences_from_file(input_file, max_batch_size)
    output_dict = {}
    total_audiopaths = []

    for batch_idx, (audiopaths, prompts) in enumerate(batches):
        print(f"Processing chunked batch {batch_idx}")
        results = generator.text_completion(
            prompts,
            max_gen_len=max_seq_len,
            temperature=temperature,
            top_p=top_p,
        )

        # 紧接着generation，调用 get_promt_last_token_embedding 方法
        # h_last_real_token_b, h_ave_real_token_b, h_last_real_token_slt, h_ave_real_token_slt, h_pca_real_token_slt, h_mat_real_token_sl = generator.get_text_prompt_token_embedding()
        # _, _, h_last_real_token_slt, _, _, _ = generator.get_text_prompt_token_embedding()
        # _, _, _, h_ave_real_token_slt, _, _ = generator.get_text_prompt_token_embedding()
        # _, _, _, _, h_pca_real_token_slt, _ = generator.get_text_prompt_token_embedding()
        _, _, _, _, _, h_mat_real_token_sl = generator.get_text_prompt_token_embedding()

        # gt_embeddings = h_last_real_token_slt.cpu()
        # gt_embeddings = h_ave_real_token_slt.cpu()
        # gt_embeddings = h_pca_real_token_slt.cpu()
        gt_embeddings = [tensor.cpu() for tensor in h_mat_real_token_sl]


        total_audiopaths.extend(audiopaths)
            
        for audiopath, prompt, result, embedding in zip(audiopaths, prompts, results, gt_embeddings):
            print(f"geting embedding for {output_file_name}:")
            print(audiopath)
            print(prompt)
            print(result)
            print(embedding[:10])
            print("\n==================================\n")

    for audiopath, embedding in zip(total_audiopaths, gt_embeddings):
        output_dict[audiopath] = embedding
    torch.save(output_dict, output_file)

if __name__ == "__main__":
    fire.Fire(main)