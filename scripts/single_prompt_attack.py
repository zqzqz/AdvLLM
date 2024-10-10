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
    attacker = PromptDosAttack(model, batch_size=16, attack_pos=args.attack_pos,logger=logger,vocab_check=args.vocab_check)
    logger.info(f"Arguments: {vars(args)}")
    for safe_cat in safe_category:
        logger.info(f"Safe Categoory:{safe_cat}")
        logger.info("**********************************")
        dataset = PromptDosDataset(data_root, num_cases=args.num_cases, safe_task_category=safe_cat, unsafe_task_category="filter_unsafe_data",task="single")
        train_dataset = dataset.train_data
        
        for case_id, case in enumerate(train_dataset):
            user_prompt, adv_prompt = case["safe_prompt"], case["unsafe_prompt"]
            print(f"user_prompt:{user_prompt}")
            # print(f"adv_prompt:{adv_prompt}")
            _, sorted_rst = attacker.test_attack(user_prompt, adv_prompt, target_response = target_response)
            is_success_flag = False
            for adv, loss in sorted_rst.items():
                new_case ={
                    "safe_prompt": user_prompt,
                    "unsafe_prompt": adv,
                }
                result_adv_prompt, rst_adv_list, train_loss_list, train_success_ratio_list = attacker.run_attack_case(new_case, num_steps=args.num_steps, target_response = target_response)
                is_success_flag, loss_list = attacker.test_attack(new_case["safe_prompt"], result_adv_prompt, target_response = target_response)
                if is_success_flag:
                    init_adv_prompt = adv
                    break

            rst = {
                "Safe Categoory":safe_cat,
                "init_adv":init_adv_prompt,
                "init_adv_length":len(init_adv_prompt),
                "result_adv":result_adv_prompt,
                "train_loss": [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in train_loss_list],  
                "loss": [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in loss_list],  
                "result_adv_length":len(result_adv_prompt),
                "result_adv_list":rst_adv_list,
            }
            with open(args.output_file, 'a') as f:
                json_line = json.dumps(rst)
                f.write(json_line + "\n")
            logger.info(f"Safe Categoory:{safe_cat}, Example {case_id}:")
            logger.info(f"User Prompt: {user_prompt}")
            logger.info(f"Adv String Init: {init_adv_prompt}, length:{len(init_adv_prompt)}")
            logger.info(f"Result:\n{result_adv_prompt}, length:{len(result_adv_prompt)}")
            logger.info(f"is_success:\n{is_success_flag}, loss:{loss_list}")
            logger.info("---------------------------------------------------------------------------")
            file_handler.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LlamaGuard universal attack.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-Guard-2-8B", help="Model name, you can choose from meta-llama/LlamaGuard-7b, meta-llama/Meta-Llama-Guard-2-8B, meta-llama/Llama-Guard-3-8B, lmsys/vicuna-7b-v1.5.")
    parser.add_argument("--log_file_name", type=str, default= "Experiment 1:Single_no_restriction_suffix", help="Log file name.")
    parser.add_argument("--attack_pos", type=str, default="suffix", help="Three options: suffix, prefix, random.")
    parser.add_argument("--output_file", type=str, default="result/Single_no_restriction_suffix.jsonl", help="Output file for the attack results.")
    parser.add_argument("--num_cases", type=int, default=10, help="Number of cases to process in the attack.")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of steps for each attack.")
    parser.add_argument("--vocab_check", type=str, default=None, help="The severity of the concealment requirement: None, Moderate, Strict")
    
    args = parser.parse_args()
    main(args)