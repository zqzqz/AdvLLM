import os
import sys
from transformers import logging as transformers_logging
import json
import argparse
import numpy as np
import traceback

transformers_logging.set_verbosity_error()
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from llm_attacks.utils import setup_logging, setup_paths, get_model_path
from llm_attacks.dataset.prompt_dataset import PromptDataset
from llm_attacks.dataset.prompt_dos_dataset import PromptDosDataset
from llm_attacks.model.llama_guard import LlamaGuard
from llm_attacks.model.vicuna import Vicuna
from llm_attacks.attack.prompt_dos_attack import PromptDosAttack
from llm_attacks.defense.smooth_llm import SmoothLLM
import matplotlib.pyplot as plt

model_path = get_model_path()
import os

workspace_root, data_root, log_root = setup_paths()

model_names = {
    "llama-guard-7b": "meta-llama/LlamaGuard-7b",
    "llama-guard-2-8b": "meta-llama/Meta-Llama-Guard-2-8B",
    "llama-guard-3-8b": "meta-llama/Llama-Guard-3-8B",
    "vicuna-7b": "lmsys/vicuna-7b-v1.5",
}

attack_pos_list = ["suffix", "prefix", "random"]


def run(model_name, attack_pos, defense_mode, perturbation_rate, input_file, output_file, log_file):
    logger, file_handler = setup_logging(log_root, log_file)
    logger.info(f"Start {input_file} {defense_mode} {perturbation_rate}")

    safe_category = ["LogicalReasoning_safe","Math_safe","Programming_safe","ReadingComprehension_safe","alpaca_safe"]
    dataset = PromptDosDataset(data_root, num_cases=20, num_universal=12, test_cases=200, safe_task_category=safe_category, unsafe_task_category="new_filter_unsafe_data", task="universal")
    test_dataset = dataset.test_data
    test_prompts = test_dataset[0]["safe_prompt"]

    if "llama" in model_name:
        model = LlamaGuard(model_id=os.path.join(model_path, model_name))
        target_response = "unsafe"
    else:
        model = Vicuna(model_id=os.path.join(model_path, model_name))
        target_response = "No"

    attacker = PromptDosAttack(model, batch_size=32, attack_pos=attack_pos, logger=logger, vocab_check="None")
    defender = SmoothLLM(attacker, config={"mode": defense_mode, "perturbation_rate": perturbation_rate}, logger=logger)

    with open(input_file, 'r') as f:
        lines = f.readlines()
        input_data = [json.loads(line) for line in lines]

    # Only do 1 case
    input_data = input_data[:1]

    result = np.ones((len(input_data), len(test_prompts))).astype(bool)

    for case_id, case_data in enumerate(input_data):
        adv_prompt = case_data["result_adv"]
        for test_id, test_prompt in enumerate(test_prompts):
            logger.info(f"case {case_id} test {test_id}")
            # test_prompt = test_prompt[:1000]
            try:
                prompt = attacker.prompt_manager.build(test_prompt, adv_prompt, target_response)
                result[case_id, test_id] = defender.run(prompt)
            except Exception as e:
                logger.error(f"Failed {e}")

    logger.info("success rate: {}".format(result.mean()))
    np.save(output_file, result)


def run_all():
    experiment_dir = "experiments"
    input_files = []
    for root, dirs, files in os.walk(experiment_dir):
        for file in files:
            if file.endswith('.jsonl'):
                input_files.append(os.path.join(root, file))

    for input_file in input_files:
        if "Experiment 3" not in input_file:
            continue
        for defense_mode in ["insert", "replace", "patch"]:
            for perturbation_rate in [0.05, 0.10]:
                log_file = input_file.split('.')[0] + f"_smooth_llm_{defense_mode}_{perturbation_rate}"
                log_file = log_file.replace("/", "__")
                output_file = input_file.split('.')[0] + f"_smooth_llm_{defense_mode}_{perturbation_rate}.npy"

                # if os.path.isfile(output_file):
                #     continue
                
                for model_key in list(model_names.keys()):
                    if model_key in input_file:
                        model_name = model_names[model_key]
                for attack_pos in attack_pos_list:
                    if attack_pos in input_file:
                        break

                run(model_name, attack_pos, defense_mode, perturbation_rate, input_file, output_file, log_file)


def test(model_name, defense_mode, perturbation_rate, output_file, log_file):
    logger, file_handler = setup_logging(log_root, log_file)
    
    task_categories = ["LogicalReasoning_safe", "Math_safe", "Programming_safe", "ReadingComprehension_safe", "alpaca_safe"]
    dataset = PromptDataset(data_root)

    if "llama" in model_name:
        model = LlamaGuard(model_id=os.path.join(model_path, model_name))
        target_response = "unsafe"
    else:
        model = Vicuna(model_id=os.path.join(model_path, model_name))
        target_response = "No"

    attacker = PromptDosAttack(model, batch_size=16, attack_pos="suffix", logger=logger)
    defender = SmoothLLM(attacker, config={"mode": defense_mode, "perturbation_rate": perturbation_rate}, logger=logger)

    result_defense = []
    gt = []
    for task_id, task in enumerate(task_categories):
        user_prompts = [p[:500] for p in dataset.data["safe"][task][:40]]
        for user_prompt in user_prompts:
            try:
                result_defense.append(defender.run(attacker.prompt_manager.build(user_prompt, "", target_response)))
            except:
                logger.error(traceback.format_exc())
        gt += [0 for i in range(len(result_defense) - len(gt))]
    user_prompts = [p[:500] for p in dataset.data["unsafe"]["unsafe_data"][:200]]
    for user_prompt in user_prompts:
        try:
            result_defense.append(defender.run(attacker.prompt_manager.build(user_prompt, "", target_response)))
        except:
            logger.error(traceback.format_exc())
    gt += [1 for i in range(len(result_defense) - len(gt))]

    data = np.asarray([result_defense, gt])
    np.save(output_file, data)


def test_all():
    for model_key, model_name in model_names.items():
        for defense_mode in ["insert", "replace", "patch"]:
            for perturbation_rate in [0.05, 0.10]:
                output_file = f"experiments/{model_key}/smooth_llm_{defense_mode}_{perturbation_rate}_test_all.npy"
                log_file = f"experiments__{model_key}__smooth_llm_{defense_mode}_{perturbation_rate}_test_all"

                if os.path.isfile(output_file):
                    continue
                test(model_name, defense_mode, perturbation_rate, output_file, log_file)


if __name__ == "__main__":
    # run_all()
    test_all()
