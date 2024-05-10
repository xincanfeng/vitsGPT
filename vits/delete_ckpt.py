import os
import re


# 定义一个函数，用于判断文件名中的数字是否可以被10,000整除
def is_not_divisible_by_number(filename):
    numbers = re.findall(r'\d+', filename)
    for num in numbers:
        if int(num) % 50000 != 0 and int(num) != 800000:
            return True
        elif int(num) > 150000:
            return True
        elif int(num) != 150000:
            return True
    return False


# 定义目标文件夹路径cdv
directory_path = 'vits/sem_vits/logs/emovdb_sem_mat_bert_text_pretrained16'


# 遍历目标文件夹中的所有文件
for filename in os.listdir(directory_path):
    full_path = os.path.join(directory_path, filename)
    if os.path.isfile(full_path) and is_not_divisible_by_number(filename):
        # 输出要删除的文件名（可选）
        print(f"Deleting {filename} ...")
        # 删除文件
        os.remove(full_path)

print("Done!")
