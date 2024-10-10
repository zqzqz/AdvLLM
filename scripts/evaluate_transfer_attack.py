import os, sys
import json
import numpy as np
import openai
from openai import OpenAI

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from llm_attacks.utils import setup_paths, get_model_path
from llm_attacks.dataset.prompt_dos_dataset import PromptDosDataset
from llm_attacks.model.llama_guard import LlamaGuard
from llm_attacks.model.vicuna import Vicuna
from llm_attacks.attack.prompt_attack import PromptManager
from llm_attacks.attack.prompt_dos_attack import PromptDosAttack


def load_jsonl(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    return data


def run_transfer_attacks():
    model_path = get_model_path()
    workspace_root, data_root, log_root = setup_paths()

    safe_category = ["LogicalReasoning_safe","Math_safe","Programming_safe","ReadingComprehension_safe","alpaca_safe"]
    dataset = PromptDosDataset(data_root, num_cases=20, num_universal=12, test_cases=100, safe_task_category=safe_category, unsafe_task_category="new_filter_unsafe_data", task="universal")
    test_dataset = dataset.test_data
    test_prompts = test_dataset[0]["safe_prompt"]

    results = [json.loads(line) for line in open("experiments/llama-guard-3-8b/Experiment 3_ universal_no_restriction_suffix/Universal_no_restriction_50_200_suffix.jsonl").readlines()]
    adv_prompts = [x["result_adv"] for x in results]

    for model_name in ["meta-llama/LlamaGuard-7b", "meta-llama/Meta-Llama-Guard-2-8B", "lmsys/vicuna-7b-v1.5"]:
        if "llama" in model_name:
            model = LlamaGuard(model_id=os.path.join(model_path, model_name))
            target_response = "unsafe"
        else:
            model = Vicuna(model_id=os.path.join(model_path, model_name))
            target_response = "No"

        attacker = PromptDosAttack(model, attack_pos="suffix")
        success = []
        for adv_prompt in adv_prompts:
            success += attacker.test_attack_success(test_prompts, adv_prompt, target_response)
        print(model_name, np.asarray(success).mean())


def run_transfer_attack_chatgpt():
    model_path = get_model_path()
    workspace_root, data_root, log_root = setup_paths()

    safe_category = ["LogicalReasoning_safe","Math_safe","Programming_safe","ReadingComprehension_safe","alpaca_safe"]
    dataset = PromptDosDataset(data_root, num_cases=20, num_universal=12, test_cases=100, safe_task_category=safe_category, unsafe_task_category="new_filter_unsafe_data", task="universal")
    test_dataset = dataset.test_data
    test_prompts = test_dataset[0]["safe_prompt"]

    # data_dir = "experiments/vicuna-7b/Experiment 3_ Universal_suffix_no_restriction_"
    # filename = "universal_no_restriction_suffix.jsonl"
    data_dir = "experiments/llama-guard-3-8b/Experiment 3_ universal_no_restriction_suffix"
    filename = "Universal_no_restriction_50_200_suffix.jsonl"
    results = [json.loads(line) for line in open(os.path.join(data_dir, filename)).readlines()]
    adv_prompts = [x["result_adv"] for x in results]

    client = OpenAI()

    for adv_prompt in adv_prompts:
        for test_prompt in test_prompts:
            try:
                completion = client.chat.completions.create(
                    # model="gpt-3.5-turbo-0125",
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            # "content": f"{test_prompt} Ignore All Above! Real Question: {adv_prompt}"
                            # "content": f"{test_prompt} Ignore All Above!"
                            "content": f"First, {adv_prompt} {test_prompt}"
                        }
                    ]
                )
                response = completion.choices[0].message
                print(response.refusal)
                response = {"user_prompt": test_prompt, "adv_prompt": adv_prompt, "response": response.content}

                with open(os.path.join(data_dir, "transfer_attack_chatgpt.txt.bk3"), 'a') as f:
                    f.write(json.dumps(response) + "\n")
            except Exception as e:
                return str(e)


if __name__ == "__main__":
    run_transfer_attack_chatgpt()