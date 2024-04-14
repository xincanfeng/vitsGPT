# Using Pretrained Models
# Different models require different model-parallel (MP) values
# --nproc_per_node 1 # 7b
# --nproc_per_node 2 # 13b
# --nproc_per_node 8 # 70b

# using llama-2-13b-chat
torchrun --nproc_per_node 2 examples/ljspeech_examples/example_chat_completion_emo_eis_sentence.py \
    --ckpt_dir llama-2-13b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 16 

# using llama-2-7b-chat
# torchrun --nproc_per_node 1 examples/ljspeech_examples/example_chat_completion_emo_eis_sentence_7b.py \
#     --ckpt_dir llama-2-7b-chat/ \
#     --tokenizer_path tokenizer.model \
#     --max_seq_len 512 --max_batch_size 16
