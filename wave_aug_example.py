from datasets import load_dataset, Dataset, load_from_disk
import soundfile as sf
import numpy as np
import librosa
#添加随机噪声
def add_noise(data):
    noise = np.random.normal(-0.1, 0.1, len(data))
    data_noise = data + noise
    return data_noise

#改变音频的速度，但不改变其音调
def stretch(data):
    rate = np.random.uniform(0.75, 1.25)
    
    input_length = len(data)
    data = librosa.effects.time_stretch(data, rate)
    if len(data)>input_length:
        return data[:input_length]
    else:
        return np.pad(data, (0, max(0, input_length - len(data))), "constant")

#改变音频的音高
def pitch_shift(data, sample_rate, steps=-3):
    
    return librosa.effects.pitch_shift(data, sample_rate, steps)

#将音频数据向左或向右随机滚动
def random_shift(data, roll_rate=0.1):
    # Roll_rate is the fraction of total length to roll
    return np.roll(data, int(len(data) * roll_rate))





train_dataset = load_from_disk("temp_datasets/en-final")

data = train_dataset[0]

speech = data["speech"]
sf.write(f'temp_audio/aug/original.wav', speech, 16000)

temp = 