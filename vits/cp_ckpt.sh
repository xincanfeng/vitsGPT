#!/bin/bash

# cp -r vits/ori_vits/logs/* /data/hf_data/Llama-VITS_data/checkpoints/ori_vits_logs/
# cp -r vits/emo_vits/logs/* /data/hf_data/Llama-VITS_data/checkpoints/emo_vits_logs/
# cp -r vits/sem_vits/logs/* /data/hf_data/Llama-VITS_data/checkpoints/sem_vits_logs/




# # 指定要搜索的根目录
# root_dir="/data/hf_data/Llama-VITS_data/checkpoints/"

# # 使用find命令查找目录名中包含"eval"的目录，并删除它们
# find "$root_dir" -type d -name 'eval' -exec rm -rf {} +
# # 输出已删除的目录列表（可选）
# echo "Deleted directories:"
# find "$root_dir" -type d -name 'eval' -print

# # 使用find命令查找目录名中包含"eval"的目录，并删除它们
# find "$root_dir" -type d -name 'G_*' -exec rm -rf {} +
# # 输出已删除的目录列表（可选）
# echo "Deleted directories:"
# find "$root_dir" -type d -name 'G_*' -print

# # 使用find命令查找文件名中包含"eval"的文件，并删除它们
# find "$root_dir" -type f -name '*eval*' -exec rm -f {} +
# # 输出已删除的文件列表（可选）
# echo "Deleted files:"
# find "$root_dir" -type f -name '*eval*' -print










# tar -cvzf /data/hf_data/Llama-VITS_data/checkpoints/ori_vits_logs.tar.gz /data/hf_data/Llama-VITS_data/checkpoints/ori_vits_logs
# tar -cvzf /data/hf_data/Llama-VITS_data/checkpoints/emo_vits_logs.tar.gz /data/hf_data/Llama-VITS_data/checkpoints/emo_vits_logs
tar -cvzf /data/hf_data/Llama-VITS_data/checkpoints/sem_vits_logs.tar.gz /data/hf_data/Llama-VITS_data/checkpoints/sem_vits_logs

# tar -xzvf /data/hf_data/Llama-VITS_data/checkpoints/ori_vits_logs.tar.gz -C /data/test_hf
