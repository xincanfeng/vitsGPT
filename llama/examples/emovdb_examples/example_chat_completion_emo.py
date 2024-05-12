# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import fire
from llama import Llama
import torch


def load_sentences_from_file(input_file: str, batch_size: int):
    """
    从给定的文件中加载句子，并按batch_size返回批次。
    文件的格式应为: ID|句子1|句子2
    """
    audiopaths = []
    sentences = []
    batches = []

    with open(input_file, "r") as file:
        for line in file:
            audiopath, sentence = line.strip().split("|")
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
    input_file: str = "datasets/EmoV_DB_bea_filtered/metadata.csv",
    output_file: str = "datasets/EmoV_DB_bea_filtered/emovdb-bea_audio_emo.txt",
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

    batches = load_sentences_from_file(input_file, max_batch_size)
    output_dict = {}
    total_audiopaths = []

    for audiopaths, sentences in batches:
        total_audiopaths.extend(audiopaths)
        # 为每个句子构建对话
        for sentence in sentences:
            dialogs = [
                [
                    {
                        "role": "system",
                        "content": "Always answer using a word among: Amused, Angry, Disgusted, Neutral, Sleepy. Or answer 'not sure' if you are not sure about the answer.",
                    },
                    {
                        "role": "user",
                        "content": f"what is the emotion of the sentence: {sentence}",
                    },
                ],
            ]

            results = generator.chat_completion(
                dialogs,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            for dialog, result in zip(dialogs, results):
                for msg in dialog:
                    print(f"{msg['role'].capitalize()}: {msg['content']}\n")
                print(
                    f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                )
                print("\n==================================\n")

            output_dict[sentence] = results

    with open(output_file, "w") as file:
        for sentence, results in output_dict.items():
            # 这里假设 result 已经是一个字符串。如果不是，请调用 str() 进行转换。
            file.write(f"{results}|{sentence}\n")


if __name__ == "__main__":
    fire.Fire(main)
