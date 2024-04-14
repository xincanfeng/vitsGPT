# Using Pretrained Models
# Different models require different model-parallel (MP) values
# --nproc_per_node 1 # 7b
# --nproc_per_node 2 # 13b
# --nproc_per_node 8 # 70b

# # using llama-2-7b 
# torchrun --nproc_per_node 1 examples/example_text_completion.py \
#     --ckpt_dir llama-2-7b/ \
#     --tokenizer_path tokenizer.model \
#     --max_seq_len 128 --max_batch_size 4

# using llama-2-13b 
torchrun --nproc_per_node 2 examples/bk/example_text_completion_emo_test.py \
    --ckpt_dir llama-2-13b/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 256 --max_batch_size 8

# # using llama-2-7b-chat
# torchrun --nproc_per_node 1 examples/example_chat_completion_emo2.py \
#     --ckpt_dir llama-2-7b-chat/ \
#     --tokenizer_path tokenizer.model \
#     --max_seq_len 512 --max_batch_size 16

# # using llama-2-13b-chat
# torchrun --nproc_per_node 2 examples/example_chat_completion_emo_eis_word.py \
#     --ckpt_dir llama-2-13b-chat/ \
#     --tokenizer_path tokenizer.model \
#     --max_seq_len 512 --max_batch_size 16
