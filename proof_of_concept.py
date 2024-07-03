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

from jiwer import wer

def calculate_wer(predictions, references):
    total_wer = 0.0
    for pred, ref in zip(predictions, references):
        total_wer += wer(ref, pred)
    average_wer = total_wer / len(predictions)
    return average_wer

asr = SLAM_ASR(
    speech_encoder_model_id ="facebook/hubert-base-ls960",
    language_model_id="openlm-research/open_llama_7b",
    # language_model_id="temp_models/rwkv-6-world-1b6",
    train_mode="adapter",
)
# load the state_dict from output/adapter_weights.pt
adapter_weight = load_file("output/checkpoint-1100/model.safetensors")
asr.load_state_dict(adapter_weight, strict=False)

import resampy
def map_to_array(batch):
    audio_data_resampled = batch["audio"]["array"]
    # audio_data_resampled = resampy.resample(batch["audio"]["array"], 48000, 16000)
    batch["speech"] = audio_data_resampled
    batch['text'] = batch["text"]
    return batch


ds = load_dataset("librispeech_asr","clean",split="validation")

ds = ds.select(range(100))
ds = ds.map(map_to_array)

print(ds)


predictions = []
references = []

# with open("temp_audio/text.txt",'w') as f:
for i in range(len(ds)):
    x = ds[i]["speech"]
    z = ds[i]["text"]
    # asr(x)
    print(f"Audio length:{len(x)}")

    output = asr.generate(x)  # causal of shape (b, seq_len, vocab_size)

    output = asr.language_tokenizer.batch_decode(output)[0]
    output = output.replace("[PAD]","")
    print(f"Predicted: {output}")
    print(f"Source:{z}")
    print("\n")
    
    predictions.append(output)
    references.append(z)
    
average_wer = calculate_wer(predictions, references)
print(f"Average WER: {average_wer}")    
    
