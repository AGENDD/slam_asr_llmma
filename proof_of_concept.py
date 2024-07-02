from datasets import load_dataset
# import soundfile as sf
import torch
from modeling.asr import SLAM_ASR
from safetensors.torch import load_file
from datasets import load_from_disk

import soundfile as sf
import numpy as np
import os
from playsound import playsound


torch.cuda.set_device(1)

asr = SLAM_ASR(
    speech_encoder_model_id ="facebook/hubert-base-ls960",
    # language_model_id="openlm-research/open_llama_3b",
    language_model_id="temp_models/rwkv-6-world-1b6",
    train_mode="adapter",
)
# load the state_dict from output/adapter_weights.pt
adapter_weight = load_file("output/rwkv/checkpoint-1200/model.safetensors")
asr.load_state_dict(adapter_weight, strict=False)

import resampy
def map_to_array(batch):
    audio_data_resampled = resampy.resample(batch["audio"]["array"], 48000, 16000)
    batch["speech"] = audio_data_resampled
    batch['text'] = batch["sentence"]
    return batch


ds = load_dataset("mozilla-foundation/common_voice_13_0","zh-CN")

ds = ds['validation'].select(range(100))
ds = ds.map(map_to_array)

# with open("temp_audio/text.txt",'w') as f:
for i in range(len(ds)):
    x = ds[i]["speech"]
    z = ds[i]["text"]
    # asr(x)
    
    
    # print(f"speech:{len(x)}")
    output = asr.generate(x)["logits"][0]  # causal of shape (b, seq_len, vocab_size)
    # print(output.shape)
    # print(f"output:{output}")
    
    token_ids = output.argmax(dim=-1)
    output = asr.language_tokenizer.decode(token_ids)
    output = output.replace("[PAD]","")
    print(f"Predicted: {output}")
    print(f"Source:{z}")
    print("\n\n")
        
        # f.write(f"Predicted: {output}\n")
        # f.write(f"Reference: {y}\n")
        # f.write(f"Source:{z}")
        # f.write("\n\n")
        
        # sf.write(f'temp_audio/temp{i}.wav', x, 16000)
        # playsound('temp.wav')
        # os.remove('temp.wav')
