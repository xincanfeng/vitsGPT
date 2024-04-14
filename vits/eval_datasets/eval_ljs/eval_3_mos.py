import os
import torch
import librosa
import datetime
import sys


# 使用 sys.argv 来获取命令行参数
method = sys.argv[1]
model = sys.argv[2]
step = sys.argv[3]

model_step_dir = f"/data/vitsGPT/vits/{method}_vits/logs/{model}/{step}/"
model_audio_folder_dir = f"{model_step_dir}model_test_wav/" 
model_mos_results_path = f"{model_step_dir}mos_results.txt"


# 确定使用的设备（CPU or GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_wavs_in_directory(directory_path, log_path):
    # 加载模型
    print("loading predictor...")
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.1.0", "utmos22_strong", trust_repo=True)
    # 将模型移动到 GPU
    predictor = predictor.to(device)

    scores = []
    with open(log_path, "w") as log_file:
        # 遍历目录中的所有WAV文件
        for filename in os.listdir(directory_path):
            if filename.endswith(".wav"):
                wav_path = os.path.join(directory_path, filename)
                
                # 加载WAV文件
                wave, sr = librosa.load(wav_path, sr=None, mono=True)
                # 将音频数据移动到 GPU
                wave_tensor = torch.from_numpy(wave).unsqueeze(0).to(device)
                
                # 获取得分
                print("calculating score...")
                score = predictor(wave_tensor, sr)
                scores.append(score.item())

                # 输出到日志文件
                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                log_file.write(f"{current_time} INFO: {filename[:-4]} {score.item()}\n")

        # 输出总文件数、平均得分和得分的方差
        print("writing log file...")
        log_file.write(f"\nTotal files processed: {len(scores)}\n")
        average = round(sum(scores)/len(scores), 4)
        variance = round(sum([(i - sum(scores)/len(scores))**2 for i in scores]) / len(scores), 4)
        log_file.write(f"Average: {average} ± {variance}\n")
        log_file.write("Successfully finished UTMOS evaluation.\n")
    
process_wavs_in_directory(model_audio_folder_dir, model_mos_results_path)

