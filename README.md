# LLM Denial-of-service Attack via Adversarial Prompts

## Overview

This repository contains tools and data for conducting Denial-of-Service (DoS) attacks on large language models (LLMs) by exploiting their safety rules. The objective is to design adversarial prompts that intentionally trigger safety mechanisms, causing models to deny service by rejecting user inputs as unsafe. This project supports universal adversarial prompt generation with various levels of universality, vocabulary restrictions, and prompt placement strategies.


## Quick Start

To set up the environment, follow these steps:

```bash
conda create -n attack python=3.10
conda activate attack
pip install -r requirements.txt
```
This is sufficient to install all dependencies and configure the environment for running the scripts.

Download data via the Google drive [link](https://drive.google.com/drive/folders/1LLDNqUyNqSstmFjiogNQK08Vfb8FFl_7?usp=sharing) and unzip it at the root of this workspace. It is a folder named `data`, containing a set of safe prompts, a set of unsafe prompts, and our custom toxic word deny-list.

You can directly run the script to reproduce all results using the code, whose result is saved to `experiments` by default:
```bash
bash run_attack.sh
```

Alternatively, please look at scripts under `scripts` and unit tests under `test` to better understand our implementation. It is an ongoing effort to improve the user interface.

## Directory Structure

Here's the structure of the repository:

```plaintext
├── data
│   ├── safe
│   ├── unsafe
|   ├── toxic_words.txt
├── llm_attacks
│   ├── attack
│   ├── dataset
│   ├── defense
│   ├── model
│   └── utils.py
├── scripts
├── test
└── third_party
```

- data/: Contains safe and unsafe prompts, and toxic words list.
- scripts/: Contains scripts to run various attacks (e.g., universal_attack.py).
- llm_attacks/: Core functionality, including the attack, dataset, and model-related code.
- test/: Contains test scripts for the project.


## Data

This repository contains both safe and unsafe prompts, organized under the `data` directory. Unsafe prompts are located in the `data/unsafe` folder and are stored in `jsonl` files.The content in each `jsonl` file is formatted as a list of dictionaries, where each dictionary has the key `"unsafe_string"`. These prompts are sourced from [Safety Prompts](https://safetyprompts.com/#harmbench). And we calculate loss using these unsafe_promot to do dos-attack and sort then we get final  dataset named `new_filter_unsafe_data.jsonl`.  You can either use our pre-filtered candidate unsafe prompts, named `new_filtered_unsafe_prompt.json`, or filter your own dataset by running:
```bash
python scripts/find_unsafe_cand.py --dataset "unsafe_data"
```

Safe prompts are found in the `data/safe` folder, also in `jsonl` files. The content in each `jsonl` file is formatted as a list of dictionaries, where each dictionary has the key `"safe_string"`.The sources for these safe prompts include:
- [AGIEval](https://github.com/ruixiangcui/AGIEval)
- [Human-Eval](https://github.com/openai/human-eval)
- [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)

For experiment with **Strict** mode Token filter a list of toxic words (`toxic_words.txt`) is used to filter adversarial prompts. You can modify this list to include additional words and run custom experiments.

## Experiments
To run the experiments, utilize the code in the `scripts` folder. Below are the scripts and their corresponding experiments:

- `single_task_attack.py` - Runs Single-task attack 
- `multi_task_attack.py` - Runs Multi-task attack

### Parameters for Universal Attack Script

Both `single_task_attack.py` and `multi_task_attack.py` script accepts several arguments to control various aspects of the experiment. Below is a description of the available parameters:

- `--model_name`: The model to use for the attack. Options include:
  - `meta-llama/LlamaGuard-7b`
  - `meta-llama/Meta-Llama-Guard-2-8B`
  - `meta-llama/Llama-Guard-3-8B`
  - `lmsys/vicuna-7b-v1.5`
  
- `--log_file_name`: The name of the log file where the attack results will be saved.
  
- `--attack_pos`: Specifies the position of the adversarial prompt relative to the safe prompt. Options include:
  - `suffix`: Attach the adversarial prompt at the end of the safe prompt.
  - `prefix`: Attach the adversarial prompt at the beginning of the safe prompt.
  - `random`: Insert the adversarial prompt at a random position within the safe prompt.

- `--output_file`: The file where the results of the adversarial attack will be saved in JSONL format.
  
- `--num_cases`: The number of cases (safe prompts) to process in the attack.

- `--num_universal`: The number of universal prompts to be used for adversarial prompt generation in each case.

- `--num_steps`: The number of iterative steps for refining the adversarial prompt during the attack.

- `--test_universal`: The number of universal prompts to be used for testing the refined adversarial prompt in each attack case.

- `--vocab_check`: Specifies the level of restriction on the generated adversarial prompt. Options include:
  - `None`: No restrictions.
  - `Moderate`: Moderate restrictions for stealthness.
  - `Strict`: Strict restrictions for stealthness.

### Usage Examples

Below is an example of how to run **Experiment 3: All-universal, No-vocab-restriction, Suffix**:
```bash
python scripts/multi_task_attack.py \
    --model_name "meta-llama/Meta-Llama-Guard-2-8B" \
    --log_file_name "Meta-Llama-Guard-2-8B/multi-task_None_suffix/log.log" \
    --attack_pos "suffix" \
    --output_file "Meta-Llama-Guard-2-8B/multi-task_None_suffix/result.jsonl" \
    --num_cases 20 \
    --num_steps 50 \
    --vocab_check None
```
Alternatively, you can simply run all attack using the run_attack.sh script for convenience:
```bash
bash run_attack.sh
```

### Log and Result Storage

The results of each experiment are stored in the `log.log` and `result.json` under corresponding `$MODEL_NAME/$TASK-NAME_VOCAB-CHECK_ATTACK-POS` directory

- **Log File**:  
  Logs for each experiment are stored in `log.log` file with the log file name specified by the `--log_file_name` parameter. The logs contain detailed information about each attack case, including the initial and final adversarial prompts, losses, success ratios, and other relevant information for debugging and analysis.

- **Result File**:  
  The actual results of each experiment are stored in `result.jsonl` file. The file name for storing these results is specified by the `--output_file` parameter. The result file contains information for each attack case, such as the initial and final adversarial prompts, their lengths, and the final success ratio. This allows for further analysis of the attack performance.

Example result file structure:
```json
{
  "id": 0,
  "init_adv": "initial adversarial string",
  "init_adv_length": 10,
  "result_adv": "final adversarial string",
  "result_adv_length": 5,
  "test_success_ratio": 0.95,
  "train_loss_list": [0.8, 0.6, 0.4],
  "train_success_ratio_list": [0.3, 0.5, 0.7],
  "result_adv_list": ["adversarial string at step 1", "adversarial string at step 2"]
}
``` 

The logs will follow the format:

```plaintext
Arguments:
Safe Category (or Test Prompt List):
********************************************************************************************************************
Example n:
User Prompt: [user_prompt]
Adv String Init: [adv_string_init], length:[len(adv_string_init)]
...
Results after each attack iteration.
...
Final Adv String: [result_adv_string], length:[len(result_adv_string)], loss: [loss(result_adv_string)]
Final Test Success Ratio:[final_test_success_ratio]
Result adv list: [adv_string after each iteration]
Train Success Ratio: [train_success_ratio_list], Train loss: [train_loss_list] 
--------------------------------------------------------------------------------------------------------------------
```
In this format, placeholders such as `[user_prompt]`, `[adv_string_init]`, and `[result_adv_string]` are replaced by the actual values used or generated during the experiment.


## References

- [AGIEval](https://github.com/ruixiangcui/AGIEval)
- [Human-Eval](https://github.com/openai/human-eval)
- [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)
- [Safety Prompts](https://safetyprompts.com/#harmbench)
- [LlamaGuard](https://github.com/meta-llama/PurpleLlama)

## Cited as

```
@article{zhang2024safeguard,
  title={Safeguard is a Double-edged Sword: Denial-of-service Attack on Large Language Models},
  author={Zhang, Qingzhao and Xiong, Ziyang and Mao, Z Morley},
  journal={arXiv preprint arXiv:2410.02916},
  year={2024}
}
```
