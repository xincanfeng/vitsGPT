# Using Pretrained Models
# Different models require different model-parallel (MP) values
# --nproc_per_node 1 # 7b
# --nproc_per_node 2 # 13b
# --nproc_per_node 8 # 70b

# using llama-2-13b 
torchrun --nproc_per_node 2 examples/ljspeech_examples/example_text_completion_emo_last.py \
    --ckpt_dir llama-2-13b/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 256 --max_batch_size 50 ;

# using llama-2-7b 
# torchrun --nproc_per_node 1 examples/ljspeech_examples/example_text_completion_emo_last_7b.py \
#     --ckpt_dir llama-2-7b/ \
#     --tokenizer_path tokenizer.model \
#     --max_seq_len 256 --max_batch_size 50