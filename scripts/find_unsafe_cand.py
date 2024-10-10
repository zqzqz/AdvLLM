"""
Script for Filtering Unsafe Prompts as Candidates for Experiments

Description:
---------
This script is designed to filter a large number of unsafe prompts and select 100 of them for use in future experiments. The final output is a JSONL file (`new_filter_unsafe_data.jsonl`), containing the filtered unsafe prompts along with their associated average loss values.

The process involves performing adversarial attacks on various `safe_prompts` and calculating the effectiveness of the `unsafe_prompts` generated by these attacks. If the `unsafe_prompt` successfully passes the attack test, it is retained for further use. Ultimately, the top 100 unsafe prompts, based on their string length, are selected.

Workflow:
---------
1. Load the dataset using the `PromptDosDataset` class with both safe and unsafe prompt categories.
2. Initialize the `LlamaGuard` model for performing adversarial attacks on the prompts.
3. For each test case, adversarial attacks are performed on the `safe_prompt`, generating an `unsafe_prompt`.
4. The `unsafe_prompt` is evaluated for success, and the average loss value is computed.
5. If the `unsafe_prompt` passes the test, it is saved to a list.
6. Sort the list of `unsafe_prompts` by their string length, select the top 100, and save them to a JSONL file.

Usage:
---------
Run this script from the terminal with:
```bash
python scripts/find_unsafe_cand.py --dataset "unsafe_data"
```
"""

import os
import sys
import json
import torch
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from llm_attacks.utils import setup_logging, setup_paths, get_model_path

workspace_root, data_root, log_root = setup_paths()
logger, file_handler = setup_logging(log_root, 'filter_unsafe_prompt')

from llm_attacks.dataset.prompt_dos_dataset import PromptDosDataset
from llm_attacks.model.llama_guard import LlamaGuard
from llm_attacks.attack.prompt_dos_attack import PromptDosAttack

model_path = get_model_path()

def main(args):
    safe_category = ["alpaca_safe","LogicalReasoning_safe","Math_safe","Programming_safe","ReadingComprehension_safe"]
    unsafe_cand = []
    dataset = PromptDosDataset(data_root, num_universal=30, safe_task_category=safe_category, unsafe_task_category=args.dataset,task="filter")
    model = LlamaGuard(model_id=os.path.join(model_path, "meta-llama/Llama-Guard-3-8B"))
    attacker = PromptDosAttack(model, batch_size=16, attack_pos="suffix",logger=logger,vocab_check=None)
    
    for case_id, case in enumerate(dataset.train_data):
        pass_test = True
        user_prompts, adv_prompt = case["safe_prompt"], case["unsafe_prompt"]
        losses = []
        for i in range(len(user_prompts)):
            user_prompt = user_prompts[i]
            print(f"user_prompt:{user_prompt}")
            print(f"adv_prompt:{adv_prompt}")
            is_success, loss = attacker.test_attack(user_prompt, adv_prompt, "unsafe")
            losses.append(loss[0].item())
            if not is_success:
                pass_test = False
                break
        if losses:
            mean_loss = sum(losses) / len(losses)
        logger.info(f"Example {case_id}:")
        logger.info(f"Adv String Init:{adv_prompt}")
        logger.info(f"Pass test:{pass_test}")
        logger.info(f"loss:{mean_loss}")
        logger.info("---------------------------------------------------------------------------")
        file_handler.flush()
        if pass_test:
            unsafe_cand.append({
                "unsafe_string":adv_prompt,
                "mean_loss": mean_loss,
                })
        torch.cuda.empty_cache()
    unsafe_cand = sorted(unsafe_cand, key=lambda x: len(x["unsafe_string"]))
    unsafe_cand =  unsafe_cand[:100]
    logger.info("Unsafe candidate prompts that passed the test:")
    for prompt in unsafe_cand:
        logger.info(prompt)

    with open('data/test.jsonl', 'w') as jsonl_file:
        for entry in unsafe_cand:
            json_line = json.dumps(entry)
            jsonl_file.write(json_line + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LlamaGuard universal attack.")
    parser.add_argument("--dataset", type=str, default="unsafe_data", help="Original dataset to filter unsafe prompts from.")

    args = parser.parse_args()
    main(args)