from datasets import load_dataset, Dataset, load_from_disk
import soundfile as sf
import numpy as np
import librosa
#添加随机噪声
def add_noise_with_fft(data):
    # 进行傅里叶变换
    data = np.array(data)
    
    fft_data = np.fft.fft(data)

    # 在频域上添加噪声
    noise = np.random.normal(0, 1, len(fft_data))
    fft_data_noise = fft_data + noise

    # 进行逆傅里叶变换
    data_noise = np.fft.ifft(fft_data_noise).real

    return data_noise

#改变音频的速度，但不改变其音调
def stretch(data, rate):
    data = np.array(data)
    
    # first = librosa.defaults.stretch
    # librosa.defaults.stretch = rate
    # input_length = len(data)
    data = librosa.effects.time_stretch(data,rate = rate)

    # librosa.defaults.stretch = first
    return data

#改变音频的音高
def pitch_shift(data, sample_rate, steps=-3):
    data = np.array(data)
    
    pitches, magnitudes = librosa.piptrack(data, sr=sample_rate)
    avg_pitch = np.average(pitches[np.nonzero(pitches)])
    
    print(avg_pitch)
    return librosa.effects.pitch_shift(data,sr= sample_rate,n_steps= steps)







train_dataset = load_from_disk("temp_datasets/en-final")

datas = [train_dataset[0],train_dataset[1000], train_dataset[2000]]

for  i,data in enumerate(datas):
    speech = data["speech"]
    sf.write(f'temp_audio/aug/{i}_original.wav', speech, 16000)

    #rate = np.random.uniform(0.75, 1.25)
    temp = stretch(speech,0.75)
    sf.write(f'temp_audio/aug/{i}_stretch_down.wav', temp, 16000)
    temp = stretch(speech,1.25)
    sf.write(f'temp_audio/aug/{i}_stretch_up.wav', temp, 16000)

    temp = pitch_shift(speech,16000, 3)
    sf.write(f'temp_audio/aug/{i}_pitch_up.wav', temp, 16000)
    temp = pitch_shift(speech,16000, -3)
    sf.write(f'temp_audio/aug/{i}_pitch_down.wav', temp, 16000)

    temp = add_noise_with_fft(speech)
    sf.write(f'temp_audio/aug/{i}_noise.wav', temp, 16000)