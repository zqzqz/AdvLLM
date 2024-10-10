import torch
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

from .prompt_attack import PromptAttack
from .utils import get_nonascii_tokens, sample_control, sample_shorter_control, sample_shorter_control_with_attention, target_loss_batch, get_filtered_cands, get_logits_by_input_ids


class PromptDosAttack(PromptAttack):
    def __init__(self, model, batch_size=32, max_fails=5, attack_pos="suffix", candidate_config={}, loss_config={}, logger=None, vocab_check=None):
        _loss_config = {
            "ce": 0.6,
            "simi": 0.1,
            "len": 0.0001,
        }
        _loss_config.update(loss_config)
        _candidate_config = {
            "topk": 56,
            "temperature": 1,
            "token_replace_n": 24,
            "token_drop_n": 8,
            "use_attention": True,
        }
        _candidate_config.update(candidate_config)
        self.vocab_check = vocab_check
        super().__init__(model, batch_size, max_fails, attack_pos, _candidate_config, _loss_config, logger, vocab_check)

        self.simi_model = SentenceTransformer("all-MiniLM-L6-v2")

    def token_filter(self):
        self.token_deny_list = get_nonascii_tokens(self.tokenizer)
        if self.vocab_check == "Moderate":
            special_symbols = []
            for i in range(128):  
                char = chr(i)
                if char.isascii() and char.isprintable() and not char.isalpha() and char != ' ':
                    special_symbols.append(char)
            special_symbol_tokens = []
            for token_id in range(self.tokenizer.vocab_size):
                token_str = self.tokenizer.decode([token_id], skip_special_tokens=True)
                if any(symbol in token_str for symbol in special_symbols):
                    special_symbol_tokens.append(token_id)
            self.token_deny_list = torch.cat([self.token_deny_list, torch.tensor(special_symbol_tokens, device=self.token_deny_list.device)])
        elif self.vocab_check == "Strict":
            file_name = "data/toxic_words.txt"
            with open(file_name, "r") as f:
                data = [line.strip() for line in f.readlines()]
            special_symbol_tokens = set()
            for word in data:
                token_ids = self.tokenizer.encode(word, add_special_tokens=False)
                special_symbol_tokens.update(token_ids) 
            for token_id in range(self.tokenizer.vocab_size):
                token_str = self.tokenizer.decode([token_id], skip_special_tokens=True).strip().lower()
                for word in data:
                    if word in token_str:
                        special_symbol_tokens.add(token_id)
            special_symbols = []
            for i in range(128):  
                char = chr(i)
                if char.isascii() and char.isprintable() and not char.isalpha() and char != ' ':
                    special_symbols.append(char)
            for token_id in range(self.tokenizer.vocab_size):
                token_str = self.tokenizer.decode([token_id], skip_special_tokens=True)
                if any(symbol in token_str for symbol in special_symbols):
                    special_symbol_tokens.add(token_id)
            self.token_deny_list = torch.cat([self.token_deny_list, torch.tensor(list(special_symbol_tokens), device=self.token_deny_list.device)])

    def loss_function(self, prompts, original_prompt):
        # Here logits and ids are on CPU device to save GPU memory.
        logits, ids = get_logits_by_input_ids(self.model,
                                              [p.input_ids for p in prompts],
                                              batch_size=self.batch_size)
        loss = self.loss_config["ce"] * target_loss_batch(logits, ids, [p.target_slice for p in prompts])
        simi = self.simi_model.similarity(
            self.simi_model.encode([original_prompt.control]),
            self.simi_model.encode([p.control for p in prompts]),
        ).squeeze(0)
        simi_loss = self.loss_config["simi"] * simi
        if simi_loss.shape[0] != loss.shape[0]:
            raise ValueError("Shape mismatch: simi_loss shape and loss shape do not match.")
        
        length_loss = torch.Tensor([len(p.control) for p in prompts])
        length_loss = self.loss_config["len"] * length_loss ** 2
        
        total_loss = loss + simi_loss + length_loss
        logging.debug(f"Total loss: {total_loss}, cross-entropy loss: {loss}, similarity loss: {simi_loss}, length loss: {length_loss}")
        return total_loss

    def generate_candidates(self, prompt, coordinate_grad, return_prompts=True):
        adv_prompt_tokens = prompt.input_ids[prompt.control_slice].to(self.device)
        new_adv_suffix_toks = sample_control(adv_prompt_tokens,
                                        coordinate_grad,
                                        self.candidate_config["token_replace_n"],
                                        topk=self.candidate_config["topk"],
                                        temp=self.candidate_config["temperature"],
                                        not_allowed_tokens=self.token_deny_list)
        new_adv_suffix, _, _ = get_filtered_cands(self.tokenizer,
                                            new_adv_suffix_toks,
                                            filter_cand=True,
                                            curr_control=prompt.control)
        if self.candidate_config["use_attention"]:
            prompts = [self.prompt_manager.build(prompt.instruction, _adv_prompt, prompt.target, prompt.insert_pos) for _adv_prompt in new_adv_suffix]
            new_adv_suffix_ids = [p.input_ids[p.control_slice] for p in prompts]
            attentions = self.get_attentions_batch(prompts)
            attentions = [att.sum(dim=0)[-1][prompts[i].control_slice] for i, att in enumerate(attentions)]
            new_adv_suffix += sample_shorter_control_with_attention(self.tokenizer,
                                                                    new_adv_suffix_ids,
                                                                    attentions,
                                                                    self.candidate_config["token_drop_n"])
        else:
            new_adv_suffix += sample_shorter_control(self.tokenizer,
                                                     new_adv_suffix,
                                                     self.candidate_config["token_drop_n"])
    
        if return_prompts:
            return [self.prompt_manager.build(prompt.instruction, _adv_prompt, prompt.target, prompt.insert_pos) for _adv_prompt in new_adv_suffix]
        else:
            return new_adv_suffix

    def select_best_candidate(self, prompts, losses, is_success):
        if isinstance(losses, torch.Tensor):
            losses = losses.detach().cpu().numpy()
        
        success_indices = np.argwhere(np.asarray(is_success) > 0).reshape(-1)
        if len(success_indices) == 0:
            return np.argmin(losses)
        
        min_index = np.argmin(losses[success_indices])
        best_new_adv_suffix_id = success_indices[min_index]
        return best_new_adv_suffix_id
    
    def get_attentions_batch(self, prompts):
        # TODO: This is very memory consuming. Set batch_size to 1 by default.
        # Overwritten this function of PromptAttack
        attentions = []
        for prompt in prompts:
            input_ids = prompt.input_ids[:prompt.target_slice.start].unsqueeze(0).to(self.model.device)
            attn_masks = torch.ones_like(input_ids).to(self.model.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attn_masks, output_attentions=True)
            attentions.append(outputs.attentions[-1][0].detach().cpu())
        return attentions
    
    def run_attack_case(self, case, num_steps=100, target_response="unsafe"):
        return self.run_attack(adv_prompt=case["unsafe_prompt"], user_prompt=case["safe_prompt"], target_response=target_response, num_steps=num_steps)
