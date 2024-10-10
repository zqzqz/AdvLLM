from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import torch


def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True
    ).to(device).eval()
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )

    # chat = [
    #     {"role": "user", "content": "Can you recommend a savory pasta recipe?"}
    # ]
    # input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    # output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    # prompt_len = input_ids.shape[-1]
    # print(tokenizer.decode(output[0], skip_special_tokens=True))
    # breakpoint()

    if 'oasst-sft-6-llama-30b' in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if 'guanaco' in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if 'llama-2' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'falcon' in tokenizer_path:
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer