# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        # "mɪsˈɛs də mˈoʊɹənskˌaɪlt θˈɔːt ðæt ˈɑːswəld,",
        "Mrs. De Mohrenschildt thought that Oswald,",
        "ðə sˈiːkɹət sˈɜːvɪs bɪlˈiːvd ðˌɐɾɪt wʌz vˈɛɹi dˈaʊtfəl ðæt ˌɛni pɹˈɛzɪdənt wʊd ɹˈaɪd ɹˈɛɡjuːlɚli ɪn ɐ vˈiəkəl wɪð ɐ fˈɪkst tˈɑːp, ˈiːvən ðˌoʊ tɹænspˈæɹənt.",
        "The Secret Service believed that it was very doubtful that any President would ride regularly in a vehicle with a fixed top, even though transparent.",
        # "Despite the intermittent rain",
        # "I believe",
        # "In the quaint town nestled between rolling hills and dense patches of cerulean bluebonnets, where children played merrily in the streets without a care in the world, and where neighbors would gather every Sunday afternoon for potlucks, sharing stories of their week, Mrs. Thompson, a sprightly old woman with a penchant for knitting colorful scarves, decided one day, much to the surprise of the townsfolk, to embark on an ambitious journey to the distant capital city, driven by her insatiable desire to witness the grand annual parade",
    ]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    # 紧接着generation，调用 get_promt_last_token_embedding 方法
    h_last_real_token_b, h_ave_real_token_b, h_last_real_token_slt, h_ave_real_token_slt, h_pca_real_token_slt, h_mat_real_token_slt = generator.get_text_prompt_token_embedding()
    # print(h_last_real_token_b.shape) 
    # print(h_ave_real_token_b)
    # print(h_last_real_token_slt)
    # print(h_ave_real_token_slt)
    print(h_mat_real_token_slt)

    # 以下已手算验证正确
    # [ 3.4414, -1.3477,  0.6040,  ..., -2.3984,  0.1274, -1.4795]
    # [ 2.3574, -1.9160,  4.0273,  ..., -1.5840,  0.7622, -0.1837]

    # 1.82
    # 1.49

    # [ 1.5918, -3.3652, -1.9580,  ..., -0.0833,  4.0117, -0.7578]
    # [ 2.3574, -1.9160,  4.0273,  ..., -1.5840,  0.7622, -0.1837]

    # 0.99
    # 1.49

    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)