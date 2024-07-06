from datasets import load_dataset, Dataset, load_from_disk
import soundfile as sf
import numpy as np
import librosa
#添加随机噪声
def add_noise_with_fft(data):
    # 进行傅里叶变换
    fft_data = np.fft.fft(data)

    # 在频域上添加噪声
    noise = np.random.normal(0, 1, len(fft_data))
    fft_data_noise = fft_data + noise

    # 进行逆傅里叶变换
    data_noise = np.fft.ifft(fft_data_noise).real

    return data_noise

#改变音频的速度，但不改变其音调
def stretch(data, rate):
    # first = librosa.defaults.stretch
    # librosa.defaults.stretch = rate
    # input_length = len(data)
    data = librosa.effects.time_stretch(data,rate)

    # librosa.defaults.stretch = first
    return data

#改变音频的音高
def pitch_shift(data, sample_rate, steps=-3):
    
    return librosa.effects.pitch_shift(data, sample_rate, steps)

#将音频数据向左或向右随机滚动
def random_shift(data, roll_rate=0.1, turn = 1):
    # Roll_rate is the fraction of total length to roll
    return np.roll(data, turn * int(len(data) * roll_rate))





train_dataset = load_from_disk("temp_datasets/en-final")

data = train_dataset[0]

speech = data["speech"]
sf.write(f'temp_audio/aug/original.wav', speech, 16000)

#rate = np.random.uniform(0.75, 1.25)
temp = stretch(speech,0.75)
sf.write(f'temp_audio/aug/stretch_down.wav', temp, 16000)
temp = stretch(speech,1.25)
sf.write(f'temp_audio/aug/stretch_up.wav', temp, 16000)

temp = pitch_shift(speech,16000, 3)
sf.write(f'temp_audio/aug/pitch_up.wav', temp, 16000)
temp = pitch_shift(speech,16000, -3)
sf.write(f'temp_audio/aug/pitch_down.wav', temp, 16000)

temp = random_shift(speech,0.1, 1)
sf.write(f'temp_audio/aug/shift_right.wav', temp, 16000)

temp = random_shift(speech,0.1, -1)
sf.write(f'temp_audio/aug/shift_left.wav', temp, 16000)

temp = add_noise_with_fft(speech)
sf.write(f'temp_audio/aug/noise.wav', temp, 16000)