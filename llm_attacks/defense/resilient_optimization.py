import torch
import logging

from llm_attacks.attack.prompt_attack import PromptSlices
from llm_attacks.attack.prompt_dos_attack import PromptDosAttack
from llm_attacks.attack.utils import target_loss_batch, get_logits_by_input_ids

# https://arxiv.org/abs/2403.13031

class AdversarialOptimizationPromptManager:
    def __init__(self, tokenizer, chat_template, mode="adv"):
        assert(mode in ["adv", "guard"])
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.mode = mode
        self.seperator = ' '
    
    def build(self, instruction, control, target, insert_pos=None):
        chat = self.chat_template.init_chat()

        user_message = self.get_user_message(instruction, control)
        self.chat_template.append_message(chat, self.chat_template.roles[0], user_message)
        tmp = self.chat_template.apply_chat_template(chat)
        toks = self.tokenizer(tmp, return_tensors='pt').input_ids[0]

        user_message_start_idx = tmp.find(user_message)
        if user_message_start_idx == -1:
            user_message_start_idx = tmp.find(user_message[:10])
        control_start_idx = tmp.find(control)
        if control_start_idx == -1:
            control_start_idx = tmp.find(control[:5])
            if control_start_idx == -1 and user_message_start_idx > 0:
                control_start_idx = user_message_start_idx + len(instruction) + len(self.seperator)
        if control_start_idx > 0 and user_message_start_idx == -1:
            user_message_start_idx = control_start_idx - len(instruction) - len(self.seperator)
        control_end_idx = control_start_idx + len(control)
        instruction_start_idx = user_message_start_idx
        if self.mode == "guard":
            instruction_end_idx = control_start_idx - len(self.seperator)
        else:
            instruction_end_idx = user_message_start_idx + len(user_message)

        adv1 = tmp[:control_start_idx]
        adv2 = tmp[:control_end_idx]
        adv_toks1 = self.tokenizer(adv1).input_ids
        adv_toks2 = self.tokenizer(adv2).input_ids
        control_slice = slice(len(adv_toks1) - 1, len(adv_toks2))

        adv3 = tmp[:instruction_start_idx]
        adv4 = tmp[:instruction_end_idx]
        adv_toks3 = self.tokenizer(adv3).input_ids
        adv_toks4 = self.tokenizer(adv4).input_ids
        instruction_slice = slice(len(adv_toks3)-1, len(adv_toks4))
        assistant_role_slice = slice(max(instruction_slice.stop, control_slice.stop), len(toks))

        if self.chat_template.name == 'llama-guard-2':
            final_toks = torch.cat((toks, self.tokenizer(target, return_tensors='pt').input_ids[0]), 0)
            if len(toks) == len(final_toks):
                target_slice = slice(len(toks)-1, len(final_toks))
            else:
                target_slice = slice(len(toks), len(final_toks))
            loss_slice = slice(target_slice.start-1, target_slice.stop-1)
        elif self.chat_template.name == 'llama-guard-3':
            final_toks = torch.cat((toks, self.tokenizer("\n\n" + target, return_tensors='pt').input_ids[0]), 0)
            target_slice = slice(len(toks) + 1, len(final_toks))
            loss_slice = slice(target_slice.start-1, target_slice.stop-1)
        elif self.chat_template.name == 'llama-guard':
            final_toks = torch.cat((toks, self.tokenizer(target, return_tensors='pt').input_ids[0]), 0)
            target_slice = slice(len(toks)+1, len(final_toks))
            loss_slice = slice(target_slice.start-2, target_slice.stop-2)
        elif self.chat_template.name == 'vicuna':
            final_toks = torch.cat((toks, self.tokenizer(target, return_tensors='pt').input_ids[0][1:]), 0)
            target_slice = slice(len(final_toks) - 1, len(final_toks))
            loss_slice = slice(target_slice.start-1, target_slice.stop-1)
        else:
            raise NotImplementedError()
        
        loss_slice = slice(target_slice.start-1, target_slice.stop-1)

        return PromptSlices(
            user_role_slice=None,
            instruction_slice=instruction_slice,
            control_slice=control_slice,
            assistant_role_slice=assistant_role_slice,
            target_slice=target_slice,
            loss_slice=loss_slice,
            instruction=instruction,
            control=control,
            target=target,
            input_ids=torch.Tensor(final_toks).type(toks.dtype),
        )

    def get_user_message(self, instruction, control):
        if self.mode == "guard":
            return f"{instruction}{self.seperator}{control}"
        elif self.mode == "adv":
            return instruction.replace("[CONTROL]", control)
        else:
            raise NotImplementedError()
        

class AdversarialOptimizationPromptAttack(PromptDosAttack):
    def __init__(self, model_class, batch_size=8, max_fails=5, mode="adv", logger=None):
        assert(mode in ["adv", "guard"])
        # No token drop; no need to shorten the control string.
        candidate_config = {
            "token_replace_n": 32,
            "token_drop_n": 0,
        }
        super().__init__(model_class=model_class,
                         batch_size=batch_size,
                         max_fails=max_fails,
                         attack_pos="suffix",
                         candidate_config=candidate_config,
                         loss_config={},
                         logger=logger)
        self.prompt_manager = AdversarialOptimizationPromptManager(self.tokenizer, self.chat_template, mode=mode)

    def loss_function(self, prompts, original_prompt):
        logits, ids = get_logits_by_input_ids(self.model,
                                              [p.input_ids for p in prompts],
                                              batch_size=self.batch_size)
        loss = target_loss_batch(logits, ids, [p.target_slice for p in prompts])
        return loss


class ResilientOptimization:
    def __init__(self, model_class, batch_size=8, logger=None):
        self.model_class = model_class
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.adv_optimizer = AdversarialOptimizationPromptAttack(model_class=model_class,
                                                                 batch_size=batch_size,
                                                                 max_fails=5,
                                                                 mode="adv",
                                                                 logger=logger)
        self.guard_optimizer = AdversarialOptimizationPromptAttack(model_class=model_class,
                                                                   batch_size=batch_size,
                                                                   max_fails=5,
                                                                   mode="guard",
                                                                   logger=logger)
        
    def run(self,
        adv_prompt="an example adversarial prompt",
        user_prompt="an example instruction",
        guard_prompt="an example guard prompt",
        adv_target_response = "an example response",
        guard_target_response = "an example response",
        num_cycles=5,
        num_steps=10
    ):
        for _ in range(num_cycles):
            adv_prompt, _, _, _ = self.adv_optimizer.run_attack(adv_prompt, f"{user_prompt} [CONTROL] {guard_prompt}", adv_target_response, num_steps=num_steps)
            guard_prompt, _, _, _ = self.guard_optimizer.run_attack(guard_prompt, f"{user_prompt} {adv_prompt}", guard_target_response, num_steps=num_steps)

        return guard_prompt