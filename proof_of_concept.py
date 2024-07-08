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
import soundfile as sf
import re
from tqdm import tqdm

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
adapter_weight = load_file("output/checkpoint-2300/model.safetensors")
asr.load_state_dict(adapter_weight, strict=False)

import resampy
def map_to_array(batch):
    audio_data_resampled = batch["audio"]["array"]
    # audio_data_resampled = resampy.resample(batch["audio"]["array"], 48000, 16000)
    batch["speech"] = audio_data_resampled
    # batch['text'] = batch["text"]
    return batch


ds1 = load_dataset("librispeech_asr","clean",split="validation")
# ds = ds.select(range(100))
ds1 = ds1.map(map_to_array,cache_file_name = "cache/cleanval.arrow")

ds2 = load_dataset("librispeech_asr","other",split="validation")
ds2 = ds2.map(map_to_array,cache_file_name = "cache/otherval.arrow")


dss = [ds1,ds2]

def contains_english(text):
    return bool(re.search('[a-zA-Z]', text))

# with open("temp_audio/text.txt",'w') as f:
for ds in dss:

    predictions = []
    references = []

    predictions_full = []
    references_full = []
    for i in tqdm(range(len(ds))):
        x = ds[i]["speech"]
        z = ds[i]["text"].lower()
        # asr(x)
        # print(f"Audio length:{len(x)/16000} s")
        
        output = asr.generate(x)  # causal of shape (b, seq_len, vocab_size)

        output = asr.language_tokenizer.batch_decode(output)[0]
        output = output.replace("<p>","")
        output = output.replace("<s>","")
        output = output.replace("</s>","")
        # print(f"Source:{z}")
        # print(f"Predicted:{output}")
        # print("\n")
        
        # f.write(f"Source:{z}\n")
        # f.write(f"Predicted:{output}\n")
        # f.write("\n")
        # sf.write(f'temp_audio/output{i}.wav', x, 16000)
        
        if(contains_english(output)):
            # print("record")
            predictions.append(output)
            references.append(z)
        predictions_full.append(output)
        references_full.append(z)
        
    average_wer = calculate_wer(predictions, references)
    average_wer_full = calculate_wer(predictions_full, references_full)
    print(f"Average WER: {average_wer}")
    print(f"Full Average WER: {average_wer_full}")
    # f.write(f"wer:{average_wer}\n")
    # f.write(f"wer full:{average_wer_full}\n")
