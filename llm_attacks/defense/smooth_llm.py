import random
import string
import numpy as np
import logging

from llm_attacks.attack.prompt_attack import PromptSlices, PromptManager

# https://arxiv.org/abs/2310.03684

class SmoothLLM():
    def __init__(self, attack_class, config={}, logger=None):
        self.config = {
            "num_samples": 31,
            "perturbation_rate": 0.05,
            "mode": "insert",
        }
        self.config.update(config)
        assert(self.config["mode"] in ["insert", "replace", "patch"])

        self.attack_class = attack_class
        self.model = attack_class.model
        self.tokenizer = attack_class.tokenizer
        self.chat_template = attack_class.chat_template
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def perturb(self, input: str):
        output = ""
        for chr in input:
            if random.random() < self.config["perturbation_rate"]:
                if self.config["mode"] == "insert":
                    output += chr + random.choice(string.ascii_lowercase)
                elif self.config["mode"] == "replace":
                    output += random.choice(string.ascii_lowercase)
                elif self.config["mode"] == "patch":
                    patch_length = min(len(output) + 1, 5)
                    output = output[:-patch_length + 1]
                    output += ''.join(random.choices(string.ascii_lowercase, k=patch_length))
            else:
                output += chr
        return output

    def run(self, prompt: PromptSlices):
        instruction, control, target = prompt.instruction, prompt.control, prompt.target
        self.logger.info(f"instruction: {instruction} control: {control} target: {target}")
        sample_prompts = []
        for _ in range(self.config["num_samples"]):
            sample_instruction = self.perturb(instruction)
            sample_control = self.perturb(control)
            sample_prompt = self.attack_class.prompt_manager.build(sample_instruction, sample_control, target)
            sample_prompts.append(sample_prompt)

        success_list = self.attack_class.check_success_batch(sample_prompts, [prompt.target], [])
        self.logger.info(f"success_list {success_list}")
        # Simple majority voting
        success = np.array(success_list).mean() > 0.5
        self.logger.info(f"success {success}")
        return success