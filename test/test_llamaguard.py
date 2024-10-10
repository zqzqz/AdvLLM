from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import os
model_path = os.environ.get("HF_MODEL_PATH")
model_path = "" if model_path is None else model_path

model_id = "meta-llama/LlamaGuard-7b"
# model_id = "meta-llama/Meta-Llama-Guard-2-8B"
# model_id = "meta-llama/Llama-Guard-3-8B"
device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, model_id))
model = AutoModelForCausalLM.from_pretrained(os.path.join(model_path, model_id), torch_dtype=dtype, device_map=device)

def moderate(chat):
    print(chat)
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    result = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    print(result)

moderate([
    {"role": "user", "content": "I forgot how to kill a process in Linux, can you help?"},
    {"role": "assistant", "content": "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate."},
])

