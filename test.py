import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_prompt(instruction, input=""):
    instruction = instruction.strip().replace('\r\n','\n').replace('\n\n','\n')
    input = input.strip().replace('\r\n','\n').replace('\n\n','\n')
    if input:
        return f"""Instruction: {instruction}

Input: {input}

Response:"""
    else:
        return f"""User: hi

Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

User: {instruction}

Assistant:"""


model = AutoModelForCausalLM.from_pretrained("temp_models/rwkv-6-world-1b6", trust_remote_code=True, torch_dtype=torch.float16).to(0)
tokenizer = AutoTokenizer.from_pretrained("temp_models/rwkv-6-world-1b6", trust_remote_code=True, padding_side='left', pad_token="<s>")

text = "介绍一下大熊猫"
prompt = generate_prompt(text)

inputs = tokenizer(prompt, return_tensors="pt").to(0)
output = model.generate(inputs["input_ids"], max_new_tokens=128, do_sample=True, temperature=1.0, top_p=0.3, top_k=0, )
print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))
