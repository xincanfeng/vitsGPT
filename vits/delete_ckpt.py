import os
import re


# 定义一个函数，用于判断文件名中的数字是否可以被10,000整除
def is_not_divisible_by_10000(filename):
    numbers = re.findall(r'\d+', filename)
    for num in numbers:
        if int(num) % 50000 != 0 and int(num) != 822000:
            return True
    return False


# 定义目标文件夹路径
directory_path = '/data/vitsGPT/vits/sem_vits/logs/ljs_sem_mat_text'


# 遍历目标文件夹中的所有文件
for filename in os.listdir(directory_path):
    full_path = os.path.join(directory_path, filename)
    if os.path.isfile(full_path) and is_not_divisible_by_10000(filename):
        # 输出要删除的文件名（可选）
        print(f"Deleting {filename} ...")
        # 删除文件
        os.remove(full_path)

print("Done!")
