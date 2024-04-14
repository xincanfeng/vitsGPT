from pydub import AudioSegment
import os

def find_zero_length_audio(folder_path):
    zero_length_files = []
    
    for file_name in os.listdir(folder_path):
        # Ensure we're working with .wav files
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            audio = AudioSegment.from_wav(file_path)
            if len(audio) == 0:
                zero_length_files.append(file_name)
    
    return zero_length_files

# Assuming the folder name is 'audio_files'
model="ori_vits"
# folder_path = f'/data/vitsGPT/vits/{model}/output_test_wav/'
folder_path = "/data/vitsGPT/vits/DUMMY1/"
zero_length_audio_files = find_zero_length_audio(folder_path)

if zero_length_audio_files:
    print(f'Found {len(zero_length_audio_files)} zero-length audio files:')
    for file_name in zero_length_audio_files:
        print(file_name)
else:
    print('No zero-length audio files found.')
