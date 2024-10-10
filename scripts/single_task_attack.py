"""
Script for Running Adversarial Prompt Attacks on LlamaGuard and Vicuna Models

Description:
---------
This script is designed to run adversarial prompt attacks on either LlamaGuard or Vicuna models to generate adversarial prompts that can cause the model to produce unsafe or undesired responses. The main purpose of the script is to evaluate how different adversarial prompts perform against these models by testing different attack positions (suffix, prefix, or random) and logging the results.

The adversarial attacks are executed on various safe prompt categories, with the goal of generating effective adversarial prompts (`unsafe_prompts`). The script refines the adversarial prompts iteratively and evaluates their effectiveness, saving the results (such as the generated prompts, their lengths, and losses) to a specified output file.

Workflow:
---------
1. Initialize the specified model (`LlamaGuard` or `Vicuna`).
2. Load safe prompts from predefined categories and attempt to generate adversarial prompts using `PromptDosAttack`.
3. For each prompt, an initial adversarial prompt is generated and iteratively refined over multiple steps.
4. The results of each adversarial attack, including the initial and refined prompts, loss values, and success flags, are saved in JSONL format.
5. Logging is performed throughout the process to track each attack's progress and outcome.

Parameters:
---------
- `--model_name`: The name of the model to use for the attack. Supported models include:
  - meta-llama/LlamaGuard-7b
  - meta-llama/Meta-Llama-Guard-2-8B
  - meta-llama/Llama-Guard-3-8B
  - lmsys/vicuna-7b-v1.5
- `--log_file_name`: The name of the log file where attack logs will be saved.
- `--attack_pos`: Specifies the position of the adversarial prompt relative to the safe prompt. Options are:
  - "suffix": Attach the adversarial prompt at the end of the safe prompt.
  - "prefix": Attach the adversarial prompt at the beginning of the safe prompt.
  - "random": Insert the adversarial prompt at a random position in the safe prompt.
- `--output_file`: The file where the results of the attacks will be saved in JSONL format.
- `--num_cases`: The number of test cases to process.
- `--num_steps`: The number of iterative steps for refining the adversarial prompt during each attack.
- `--vocab_check`: Optional parameter to specify the level of restriction on the generated adversarial prompt. Options are:
  - None: No restrictions.
  - Moderate: Moderate level of concealment requirements.
  - Strict: Strict concealment requirements.

Example Usage:
---------
Run the script from the terminal using the following command:
```bash
python scripts/single_task_attack.py --model_name "meta-llama/Meta-Llama-Guard-2-8B" --log_file_name "Experiment_1" --attack_pos "suffix" --output_file "result/Single_task_no_restriction_suffix.jsonl" --num_cases 10 --num_steps 50 --vocab_check None
```
"""
import os
import sys
from transformers import logging as transformers_logging
import json
import argparse
import torch

transformers_logging.set_verbosity_error()
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from llm_attacks.utils import setup_logging, setup_paths, get_model_path
from llm_attacks.dataset.prompt_dos_dataset import PromptDosDataset
from llm_attacks.model.llama_guard import LlamaGuard
from llm_attacks.attack.prompt_dos_attack import PromptDosAttack
from llm_attacks.model.vicuna import Vicuna

model_path = get_model_path()

def main(args):
    workspace_root, data_root, log_root = setup_paths()
    logger, file_handler = setup_logging(log_root, args.log_file_name)
    safe_category = ["LogicalReasoning_safe","Math_safe","Programming_safe","ReadingComprehension_safe","alpaca_safe"]
    if "llama" in args.model_name:
        model = LlamaGuard(model_id=os.path.join(model_path, args.model_name))
        target_response = "unsafe"
    elif "vicuna" in args.model_name:
        model = Vicuna(model_id=os.path.join(model_path, args.model_name))
        target_response = "No"
    else:
        raise NotImplementedError()
    attacker = PromptDosAttack(model, batch_size=8, attack_pos=args.attack_pos,logger=logger,vocab_check=args.vocab_check)
    logger.info(f"Arguments: {vars(args)}")
    for safe_cat in safe_category:
        result = []
        logger.info(f"Safe Categoory:{safe_cat}")
        logger.info("**********************************")
        # num_universal represents the number of safe_prompt calculate together in one time
        dataset = PromptDosDataset(data_root, num_cases=args.num_cases, num_universal=args.num_universal, test_cases = args.test_universal, safe_task_category=safe_cat, unsafe_task_category="new_filter_unsafe_data",task="task_universal")
        train_dataset = dataset.train_data
        test_dataset = dataset.test_data
        test_prompt = test_dataset[0]["safe_prompt"]
        logger.info(f"Test Prompt List:\n{test_prompt}")
        logger.info("**********************************")

        for case_id, case in enumerate(train_dataset):
            user_prompt, adv_prompt = case["safe_prompt"], case["unsafe_prompt"]
            # print(f"user_prompt:{user_prompt}")
            print(f"adv_prompt:{adv_prompt}")
            logger.info(f"Example {case_id}:")
            logger.info(f"Init User Prompt: {user_prompt}")
            _, sorted_rst = attacker.test_attack(user_prompt, adv_prompt, target_response = target_response)
            init_adv_prompt = next(iter(sorted_rst))
            new_case ={
                    "safe_prompt": user_prompt,
                    "unsafe_prompt": init_adv_prompt,
                }
            result_adv_prompt, rst_adv_list, train_loss_list, train_success_ratio_list = attacker.run_attack_case(new_case, num_steps=args.num_steps,target_response = target_response)
            train_is_success, loss_list = attacker.test_attack(new_case["safe_prompt"], result_adv_prompt, "unsafe") 
            train_success_ratio = sum(train_is_success) / len(train_is_success)
        
            test_is_success, _ = attacker.test_attack(test_prompt,  result_adv_prompt, target_response = target_response)
            test_success_ratio = sum(test_is_success) / len(test_is_success)
        
            rst = {
                "Safe Categoory":safe_cat,
                "init_adv":init_adv_prompt,
                "init_adv_length":len(init_adv_prompt),
                "result_adv":result_adv_prompt,
                "result_adv_length":len(result_adv_prompt),
                "train_success_ratio": train_success_ratio,
                "test_success_ratio":test_success_ratio,
                "result_adv_list":rst_adv_list,
                "train_loss": [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in train_loss_list],  
            }
            result.append(rst)
            with open(args.output_file, 'a') as f:
                json_line = json.dumps(rst)
                f.write(json_line + "\n")
            logger.info(f"Adv String Init: {adv_prompt}, length:{len(adv_prompt)}")
            logger.info(f"Final Adv String:{result_adv_prompt}, length:{len(result_adv_prompt)}, loss:{train_loss_list[-1]}")
            logger.info(f"Final Test Success Ratio:{test_success_ratio}")
            logger.info(f"Final Train Success Ratio:{train_success_ratio}")
            logger.info(f"Result adv list:\n{rst_adv_list}")
            logger.info(f"Train Success Ratio:{train_success_ratio_list} Train loss:{train_loss_list}")
            logger.info("---------------------------------------------------------------------------")
            file_handler.flush()
        init_adv_lengths = [len(rst["init_adv"]) for rst in result]
        avg_init_adv_length = sum(init_adv_lengths) / len(init_adv_lengths)
        result_adv_lengths = [len(rst["result_adv"]) for rst in result]
        avg_result_adv_length = sum(result_adv_lengths) / len(result_adv_lengths)
        test_success_ratios = [rst["test_success_ratio"] for rst in result]
        avg_success_ratio = sum(test_success_ratios) / len(test_success_ratios)
        logger.info(f"{safe_cat}:Average init_adv length: {avg_init_adv_length:.2f}")
        logger.info(f"{safe_cat}:Average result_adv length: {avg_result_adv_length:.2f}")
        logger.info(f"{safe_cat}:Average Test success_ratio: {avg_success_ratio:.2f}")
        torch.cuda.empty_cache()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LlamaGuard universal attack.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-Guard-2-8B", help="Model name, you can choose from meta-llama/LlamaGuard-7b, meta-llama/Meta-Llama-Guard-2-8B, meta-llama/Llama-Guard-3-8B, lmsys/vicuna-7b-v1.5.")
    parser.add_argument("--log_file_name", type=str, default= "Experiment 2:Task_universal_no_restriction_suffix", help="Log file name.")
    parser.add_argument("--attack_pos", type=str, default="suffix", help="Three options: suffix, prefix, random.")
    parser.add_argument("--output_file", type=str, default="result/Task_universal_no_restriction_suffix.jsonl", help="Output file for the attack results.")
    parser.add_argument("--num_cases", type=int, default=10, help="Number of cases to process in the attack.")
    parser.add_argument("--num_universal", type=int, default=12, help="Number of universal prompts to calculate together in one time.")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of steps for each attack.")
    parser.add_argument("--test_universal", type=int, default=30, help="Number of universal prompts to test for each attack.")
    parser.add_argument("--vocab_check", type=str, default=None, help="The severity of the concealment requirement: None, Moderate, Strict")

    args = parser.parse_args()
    main(args)