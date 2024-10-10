import os, sys
import json
import pickle
import numpy as np
import traceback
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from llm_attacks.utils import setup_paths, setup_logging, get_model_path
from llm_attacks.dataset.prompt_dataset import PromptDataset
from llm_attacks.dataset.prompt_dos_dataset import PromptDosDataset
from llm_attacks.model.llama_guard import LlamaGuard
from llm_attacks.model.vicuna import Vicuna
from llm_attacks.attack.prompt_attack import PromptManager
from llm_attacks.attack.prompt_dos_attack import PromptDosAttack


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


def load_jsonl(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    return data


def run_iterations():
    from llm_attacks.utils import setup_paths, get_model_path
    from llm_attacks.dataset.prompt_dos_dataset import PromptDosDataset
    from llm_attacks.model.llama_guard import LlamaGuard
    from llm_attacks.attack.prompt_attack import PromptManager
    from llm_attacks.attack.prompt_dos_attack import PromptDosAttack

    model_path = get_model_path()

    workspace_root, data_root, log_root = setup_paths()
    safe_category = ["LogicalReasoning_safe","Math_safe","Programming_safe","ReadingComprehension_safe","alpaca_safe"]

    dataset = PromptDosDataset(data_root, num_cases=20, num_universal=12, test_cases=100, safe_task_category=safe_category, unsafe_task_category="new_filter_unsafe_data", task="universal")
    test_dataset = dataset.test_data
    test_prompt = test_dataset[0]["safe_prompt"]
    model = LlamaGuard(model_id=os.path.join(model_path, "meta-llama/Llama-Guard-3-8B"))
    attacker = PromptDosAttack(model, batch_size=32, attack_pos="suffix", logger=None)

    results = [json.loads(line) for line in open("experiments/llama-guard-3-8b/Experiment 3_ universal_no_restriction_suffix/Universal_no_restriction_50_200_suffix.jsonl").readlines()]
    adv_prompts = results[0]["result_adv_list"]

    success_ratio_list = []
    for result_adv_prompt in adv_prompts:
        final_is_success = attacker.test_attack_success(test_prompt, result_adv_prompt, "unsafe")
        final_success_ratio = np.asarray(final_is_success).mean()
        print(final_success_ratio)
        success_ratio_list.append(final_success_ratio)
    
    print(success_ratio_list)


def run_length():

    def run_length_thread(model_name, attack_pos, input_file, output_file, log_file):
        logger, file_handler = setup_logging(log_root, log_file)
        logger.info(f"Start {input_file}")
        
        task_categories = ["LogicalReasoning_safe", "Math_safe", "Programming_safe", "ReadingComprehension_safe", "alpaca_safe"]
        dataset = PromptDataset(data_root)

        if "llama" in model_name:
            model = LlamaGuard(model_id=os.path.join(model_path, model_name))
            target_response = "unsafe"
        else:
            model = Vicuna(model_id=os.path.join(model_path, model_name))
            target_response = "No"

        attacker = PromptDosAttack(model, batch_size=32, attack_pos=attack_pos, logger=logger)
        results = load_jsonl(input_file)

        success = []
        length = []
        category = []
        for task_id, task in enumerate(task_categories):
            user_prompts = dataset.data["safe"][task][:192]
            for result in results:
                adv_prompt = result["result_adv"]
                try:
                    success += attacker.test_attack_success(user_prompts, adv_prompt, target_response)
                    length += [len(prompt) for prompt in user_prompts]
                    category += [task_id for prompt in user_prompts]
                except:
                    traceback.format_exc()
        data = np.asarray([success, length, category])
        np.save(output_file, data)

    experiment_dir = "experiments"
    input_files = []
    for root, dirs, files in os.walk(experiment_dir):
        for file in files:
            if file.endswith('.jsonl'):
                input_files.append(os.path.join(root, file))

    for input_file in input_files:
        if "Experiment 7" not in input_file and "Experiment 8" not in input_file:
            continue
        log_file = input_file.split('.')[0] + f"_test_all"
        log_file = log_file.replace("/", "__")
        output_file = input_file.split('.')[0] + f"_test_all.npy"

        if os.path.isfile(output_file):
            continue
        
        for model_key in list(model_names.keys()):
            if model_key in input_file:
                model_name = model_names[model_key]
        for attack_pos in attack_pos_list:
            if attack_pos in input_file:
                break
        try:
            run_length_thread(model_name, attack_pos, input_file, output_file, log_file)
        except KeyboardInterrupt:
            break
        except:
            traceback.format_exc()


if __name__ == "__main__":
    run_length()