"""
Script for Running Universal Adversarial Attacks on LlamaGuard and Vicuna Models

Description:
---------
This script performs universal adversarial attacks on LlamaGuard and Vicuna models using predefined safe prompts from various categories. The goal is to generate and iteratively refine adversarial prompts (`unsafe_prompts`) that lead the model to generate unsafe or undesired responses. 

The process involves:
1. Loading the specified model (LlamaGuard or Vicuna).
2. Loading a dataset of safe and unsafe prompts.
3. Iteratively optimizing adversarial prompts through multiple attack steps.
4. Testing the effectiveness of these adversarial prompts against test universal prompts.
5. Logging and saving the results, including initial and final adversarial prompts, success ratios, and loss values, to a JSONL file.

The script supports different attack positions (suffix, prefix, random) and can apply varying levels of stealthness checks to the generated adversarial prompts (None, Moderate, Strict, Toxic words).

Workflow:
---------
1. Parse command-line arguments to configure the model, attack type, output file, etc.
2. Load the dataset using `PromptDosDataset`.
3. Initialize the specified model (LlamaGuard or Vicuna).
4. For each case, generate adversarial prompts and refine them through multiple iterations.
5. Test the refined prompts against a set of universal test prompts.
6. Log the attack details (e.g., success ratios, losses) and save the results to a JSONL file.

Parameters:
---------
- `--model_name`: The model to use for the attack. Options include:
  - meta-llama/LlamaGuard-7b
  - meta-llama/Meta-Llama-Guard-2-8B
  - meta-llama/Llama-Guard-3-8B
  - lmsys/vicuna-7b-v1.5
- `--log_file_name`: The name of the log file where the attack results will be saved.
- `--attack_pos`: Specifies the attack's position. Options include:
  - suffix: Attach the adversarial prompt at the end of the safe prompt.
  - prefix: Attach the adversarial prompt at the beginning of the safe prompt.
  - random: Insert the adversarial prompt at a random position.
- `--output_file`: The file where the results of the adversarial attack will be saved in JSONL format.
- `--num_cases`: The number of cases to process in the attack.
- `--num_universal`: The number of universal prompts to be used for attack generation in each case.
- `--num_steps`: The number of iterative steps for refining the adversarial prompt.
- `--test_universal`: The number of universal prompts to be used for testing each attack case.
- `--vocab_check`: Specifies the level of restriction on the generated adversarial prompt. Options include:
  - None: No restrictions.
  - Moderate: Moderate restrictions for stealthness.
  - Strict: Strict restrictions for stealthness.

Example Usage:
---------
Run the script from the terminal using the following command:
```bash
python scripts/multi_task_attack.py --model_name "meta-llama/Meta-Llama-Guard-2-8B" --log_file_name "Experiment_3" --attack_pos "suffix" --output_file "result/Universal_no_restriction_suffix.jsonl" --num_cases 20 --num_universal 12 --num_steps 50 --test_universal 200 --vocab_check "None"
```
"""

import os
import sys
from transformers import logging as transformers_logging
import json
import argparse

transformers_logging.set_verbosity_error()
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from llm_attacks.utils import setup_logging, setup_paths, get_model_path
from llm_attacks.dataset.prompt_dos_dataset import PromptDosDataset
from llm_attacks.model.llama_guard import LlamaGuard
from llm_attacks.attack.prompt_dos_attack import PromptDosAttack
import matplotlib.pyplot as plt
from llm_attacks.model.vicuna import Vicuna

model_path = get_model_path()
import torch
import os
os.environ['HF_HOME'] = "/scratch/oymak_owned_root/oymak_owned1/xziyang/cache/huggingface/"

def main(args):
    workspace_root, data_root, log_root = setup_paths()
    logger, file_handler = setup_logging(log_root, args.log_file_name)
    safe_category = ["LogicalReasoning_safe","Math_safe","Programming_safe","ReadingComprehension_safe","alpaca_safe"]
    
    dataset = PromptDosDataset(data_root, num_cases=args.num_cases, num_universal=args.num_universal, test_cases = args.test_universal, safe_task_category=safe_category, unsafe_task_category="new_filter_unsafe_data",task="universal")
    train_dataset = dataset.train_data
    test_dataset = dataset.test_data
    test_prompt = test_dataset[0]["safe_prompt"]
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Test Prompt List:\n{test_prompt}")
    logger.info("---------------------------------------------------------------------------")
    if "llama" in args.model_name:
        model = LlamaGuard(model_id=os.path.join(model_path, args.model_name))
        target_response = "unsafe"
    elif "vicuna" in args.model_name:
        model = Vicuna(model_id=os.path.join(model_path, args.model_name))
        target_response = "No"
    else:
        raise NotImplementedError()
    attacker = PromptDosAttack(model, batch_size=16, attack_pos=args.attack_pos,logger=logger,vocab_check=args.vocab_check)
    result = []
    output_file = args.output_file


    for case_id, case in enumerate(train_dataset):
        user_prompt, adv_prompt = case["safe_prompt"], case["unsafe_prompt"]
        _, sorted_rst = attacker.test_attack(user_prompt, adv_prompt, target_response = target_response)
        print(f"sorted_rst:{sorted_rst}")
        adv_prompt = next(iter(sorted_rst))
        print(f"adv_prompt:{adv_prompt}")
        logger.info(f"Example {case_id}:")
        logger.info(f"User Prompt: {user_prompt}")
        logger.info(f"Adv String Init: {adv_prompt}, length:{len(adv_prompt)}")
        new_case ={
            "safe_prompt": user_prompt,
            "unsafe_prompt": adv_prompt,
        }
        result_adv_prompt, rst_adv_list, train_loss_list, train_success_ratio_list = attacker.run_attack_case(new_case, num_steps=args.num_steps,target_response = target_response)
        final_is_success, _ = attacker.test_attack(test_prompt,  result_adv_prompt, target_response = target_response)
        final_success_ratio = sum(final_is_success) / len(final_is_success)

        rst = {
            "id": case_id,
            "init_adv":adv_prompt,
            "init_adv_length":len(adv_prompt),
            "result_adv":result_adv_prompt,
            "result_adv_length":len(result_adv_prompt),
            "test_success_ratio":final_success_ratio,
            "train_loss_list": train_loss_list,
            "train_success_ratio_list": train_success_ratio_list,
            "result_adv_list":rst_adv_list,
        }

        result.append(rst)
        with open(output_file, 'a') as f:
            json_line = json.dumps(rst)
            f.write(json_line + "\n")
        logger.info("***************************************************************************")
        logger.info(f"Adv String Init: {adv_prompt}, length:{len(adv_prompt)}")
        logger.info(f"Final Adv String:{result_adv_prompt}, length:{len(result_adv_prompt)}, loss:{train_loss_list[-1]}")
        logger.info(f"Final Test Success Ratio:{final_success_ratio}")
        logger.info(f"Result adv list:\n{rst_adv_list}")
        logger.info(f"Train Success Ratio:{train_success_ratio_list} Train loss:{train_loss_list}")
        logger.info("---------------------------------------------------------------------------")
        file_handler.flush()
        torch.cuda.empty_cache()

    init_adv_lengths = [len(rst["init_adv"]) for rst in result]
    avg_init_adv_length = sum(init_adv_lengths) / len(init_adv_lengths)
    result_adv_lengths = [len(rst["result_adv"]) for rst in result]
    avg_result_adv_length = sum(result_adv_lengths) / len(result_adv_lengths)
    test_success_ratios = [rst["test_success_ratio"] for rst in result]
    avg_success_ratio = sum(test_success_ratios) / len(test_success_ratios)
    logger.info(f"Average init_adv length: {avg_init_adv_length:.2f}")
    logger.info(f"Average result_adv length: {avg_result_adv_length:.2f}")
    logger.info(f"Average Test success_ratio: {avg_success_ratio:.2f}")
    print(f"Result saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LlamaGuard universal attack.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-Guard-2-8B", help="Model name, you can choose from meta-llama/LlamaGuard-7b, meta-llama/Meta-Llama-Guard-2-8B, meta-llama/Llama-Guard-3-8B, lmsys/vicuna-7b-v1.5.")
    parser.add_argument("--log_file_name", type=str, default= "Experiment 3:Universal attack_with_strict_check", help="Saved log file name.")
    parser.add_argument("--attack_pos", type=str, default="suffix", help="Three options: suffix, prefix, random.")
    parser.add_argument("--output_file", type=str, default="result/Universal_no_restriction_suffix.jsonl", help="Output file for the attack results.")
    parser.add_argument("--num_cases", type=int, default=20, help="Number of cases to process in whole attack.")
    parser.add_argument("--num_universal", type=int, default=12, help="Number of universal prompts to calculate together in one case.")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of iters for each attack case.")
    parser.add_argument("--test_universal", type=int, default=200, help="Number of universal prompts to test for each attack case.")
    parser.add_argument("--vocab_check", type=str, default="None", help="The severity of the stealthness requirement: None, Moderate, Strict")

    args = parser.parse_args()
    main(args)