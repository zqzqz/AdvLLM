import gc
import torch
import numpy as np
from tqdm import tqdm
import logging
import torch.nn as nn
import torch.nn.functional as F
import random
import copy

from .utils import token_gradients, get_embedding_matrix
from collections import Counter


class PromptSlices:
    def __init__(self, user_role_slice=None, instruction_slice=None, control_slice=None, assistant_role_slice=None,
                 target_slice=None, loss_slice=None, instruction=None, control=None,
                 target=None, input_ids=None, insert_pos=None):
        self.user_role_slice = user_role_slice
        self.instruction_slice = instruction_slice
        self.control_slice = control_slice
        self.assistant_role_slice = assistant_role_slice
        self.target_slice = target_slice
        self.loss_slice = loss_slice
        self.instruction = instruction
        self.control = control
        self.target = target
        self.input_ids = input_ids
        self.insert_pos = insert_pos


class PromptManager:
    def __init__(self, tokenizer, chat_template, mode="suffix"):
        assert(mode in ["suffix", "prefix", "random"])
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.mode = mode
        self.seperator = ' '

    @staticmethod
    def find_subsequence_indices(sequence, subsequence):
        # TODO: The best complexity should be O(n log n)
        # NOTE: Not used
        seq_len = len(sequence)
        sub_len = len(subsequence)
        for i in range(seq_len - sub_len + 1):
            if torch.equal(sequence[i:i + sub_len], subsequence):
                return i
        return -1
    
    def build(self, instruction, control, target, insert_pos=None):
        chat = self.chat_template.init_chat()
        
        user_message, insert_pos = self.get_user_message(instruction, control, insert_pos)
        self.chat_template.append_message(chat, self.chat_template.roles[0], user_message)
        tmp = self.chat_template.apply_chat_template(chat)
        toks = self.tokenizer(tmp, return_tensors='pt').input_ids[0]

        user_message_start_idx = tmp.find(user_message)
        if user_message_start_idx == -1:
            user_message_start_idx = tmp.find(user_message[:10])
        control_start_idx = tmp.find(control)
        if control_start_idx == -1:
            control_start_idx = tmp.find(control[:5])
        control_end_idx = control_start_idx + len(control)
        if self.mode == "prefix":
            if control_start_idx == -1 and user_message_start_idx > 0:
                control_start_idx = user_message_start_idx 
            elif control_start_idx > 0 and user_message_start_idx == -1:
                user_message_start_idx = control_start_idx
            control_end_idx = control_start_idx + len(control)
            instruction_start_idx = control_end_idx + len(self.seperator)
        else:
            instruction_start_idx = user_message_start_idx
        if self.mode == "suffix":
            if control_start_idx == -1 and user_message_start_idx > 0:
                control_start_idx = user_message_start_idx + len(instruction) + len(self.seperator)
            control_end_idx = control_start_idx + len(control)
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
        elif self.chat_template.name == 'llama-guard-3':
            final_toks = torch.cat((toks, self.tokenizer("\n\n" + target, return_tensors='pt').input_ids[0]), 0)
            target_slice = slice(len(toks) + 1, len(final_toks))
        elif self.chat_template.name == 'llama-guard':
            final_toks = torch.cat((toks, self.tokenizer(target, return_tensors='pt').input_ids[0]), 0)
            target_slice = slice(len(toks), len(final_toks) - 1)
        elif self.chat_template.name == 'vicuna':
            final_toks = torch.cat((toks, self.tokenizer(target, return_tensors='pt').input_ids[0][1:]), 0)
            target_slice = slice(len(final_toks) - 1, len(final_toks))
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
            insert_pos=insert_pos,
        )

    def get_user_message(self, instruction, control, insert_pos=None):
        if self.mode == "prefix":
            return f"{control}{self.seperator}{instruction}", 0
        elif self.mode == "suffix":
            return f"{instruction}{self.seperator}{control}", len(instruction)
        elif self.mode == "random":
            if insert_pos is None:
                insert_pos = random.randint(0, len(instruction))
            return "{}{}{}{}{}".format(instruction[:insert_pos], self.seperator, control, self.seperator, instruction[insert_pos:]), insert_pos
    

class PromptAttack:
    def __init__(self, model_class, batch_size=32, max_fails=5, attack_pos="suffix", candidate_config={}, loss_config={}, logger=None, vocab_check=None):
        self.attack_pos = attack_pos
        self.candidate_config = {
            # The adversarial attack is successful if the adv string can work on {thres} of test cases.
            "universal_success_thres": 0.6,
        }
        self.candidate_config.update(candidate_config)
        self.loss_config = {}
        self.loss_config.update(loss_config)

        self.model_class = model_class
        self.model = model_class.model
        self.name = model_class.model_id
        self.tokenizer = model_class.tokenizer
        self.chat_template = model_class.chat_template
        self.device = self.model_class.device
        self.batch_size = batch_size
        self.max_fails = max_fails

        self.token_deny_list = None
        self.token_allow_list = None
        self.token_filter()

        self.batch_size = batch_size

        self.prompt_manager = PromptManager(self.tokenizer, self.chat_template, mode=self.attack_pos)
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.vocab_check = vocab_check

    """Definition of allow list and deny list of tokens, used when generating condaidates.
        Should be override function by child classes

        No parameters, no retunns.
        Sets self.token_deny_list and self.token_allow_list
    """
    def token_filter(self):
        raise NotImplementedError()

    """Definition of loss function representing the attack effectiveness.
        Should be override function by child classes

        Parameters: TODO
        Returns: TODO
    """
    def loss_function(self, prompts, original_prompt):
        raise NotImplementedError()
    
    """Defining how the attack search candidate of adversarial prompts.
        Should be override function by child classes

        Parameters: TODO
        Returns:
        list(PromptSlices or str): A list of candidates.
    """
    def generate_candidates(self, prompt, coordinate_grad, return_prompts=True):
        raise NotImplementedError()
    
    """Select the best candidate from a set of candidates. Using metrics like
        loss values and success flags.

        Parameters: TODO
        Returns:
        int: The best candidate index.
    """
    def select_best_candidate(self, prompts, losses, is_success):
        raise NotImplementedError()

    """A function determining whether the attack is successful.
        Should be override function by child classes

        Parameters: TODO
        Returns:
        list(bool): Boolean flag
    """
    def check_success(self, prompt, test_substr, neg_test_substr=[]):
        return self.check_success_batch([prompt], test_substr, neg_test_substr)[0]
    
    def check_success_batch(self, prompts, test_substr, neg_test_substr=[]):
        output_ids = self.generate_batch(prompts)
        success = []
        for prompt_id, prompt in enumerate(prompts):
            gen_str = self.tokenizer.decode(output_ids[prompt_id]).strip()
            if len(test_substr) > 0:
                success.append(any([prefix in gen_str for prefix in test_substr]))
            else:
                success.append(not any([prefix in gen_str for prefix in neg_test_substr]))
        logging.debug(f"check_success responses: {success}")
        return success
    
    def generate(self, prompt, max_new_tokens=8):
        return self.generate_batch([prompt], max_new_tokens)[0]

    def generate_batch(self, prompts, max_new_tokens=8):
        gen_config = self.model.generation_config
        gen_config.max_new_tokens = max_new_tokens

        max_prompt_length = np.array([prompt.assistant_role_slice.stop for prompt in prompts]).max()
        output_ids = torch.zeros(len(prompts), max_new_tokens).type(prompts[0].input_ids.dtype)
        
        for i in range(0, len(prompts), self.batch_size):
            input_ids = torch.zeros(min(self.batch_size, len(prompts) - i), max_prompt_length).type(prompts[0].input_ids.dtype)
            for prompt_id, prompt in enumerate(prompts[i:i+self.batch_size]):
                input_ids[prompt_id, -prompt.assistant_role_slice.stop:] = prompt.input_ids[:prompt.assistant_role_slice.stop]
            input_ids = input_ids.to(self.model.device)
            attn_masks = (input_ids > 0).int().to(self.model.device)
            batch_output_ids = self.model.generate(input_ids,
                                    attention_mask=attn_masks,
                                    generation_config=gen_config,
                                    pad_token_id=self.tokenizer.pad_token_id)
            output_ids[i:i+self.batch_size, :batch_output_ids.shape[1] - max_prompt_length] = batch_output_ids[:min(self.batch_size, len(prompts) - i), max_prompt_length:]
        return output_ids
    
    def get_attentions(self, prompt):
        return self.get_attentions_batch([prompt])[0]

    def get_attentions_batch(self, prompts):
        max_prompt_length = np.array([prompt.target_slice.start for prompt in prompts]).max()
        batch_size = len(prompts)
        input_ids = torch.zeros(batch_size, max_prompt_length).type(prompts[0].input_ids.dtype)
        for prompt_id, prompt in enumerate(prompts):
            input_ids[prompt_id, -prompt.target_slice.start:] = prompt.input_ids[:prompt.target_slice.start]
        input_ids = input_ids.to(self.model.device)
        attn_masks = (input_ids > 0).int().to(self.model.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attn_masks, output_attentions=True)
        attentions = outputs.attentions
        attentions = attentions[-1].detach().cpu()
        return attentions
    
    def get_gradients(self, prompt):
        input_ids = prompt.input_ids.to(self.device)
        coordinate_grad = token_gradients(self.model,
                                        input_ids,
                                        prompt.control_slice, 
                                        prompt.target_slice, 
                                        prompt.loss_slice)
        return coordinate_grad

    def run_attack(self,
        adv_prompt="an example adversarial prompt",
        user_prompt="an example instruction",
        target_response = "an example response",
        num_steps=100,
    ):
        # The attack may acquire a list of instructions and targets, in case we need a universal attack.
        assert(len(adv_prompt) > 0 and len(user_prompt) > 0 and len(target_response) > 0)

        # Convert single string to a list
        user_prompts = [user_prompt] if isinstance(user_prompt, str) else user_prompt

        original_prompts = [self.prompt_manager.build(user_prompt, adv_prompt, target_response) for user_prompt in user_prompts]
        prompts = [copy.deepcopy(original_prompt) for original_prompt in original_prompts]
        best_adv_prompt = adv_prompt
        rst_adv_prompt = best_adv_prompt

        fail_count = 0
        word_length = get_embedding_matrix(self.model).shape[0]
        rst_adv = []
        best_loss_list = []
        best_success_rate_list = []
        for i in tqdm(range(num_steps), desc="Processing steps"):
            # Compute Coordinate Gradient
            self.logger.debug(f"Iteration {i}: Generating grad ...")
            max_control_length = max([prompt.control_slice.stop - prompt.control_slice.start for prompt in prompts])
            if max_control_length <= 0:
                self.logger.error(f"Iteration {i}: Invalid max_control_length {max_control_length}. Skipping iteration.")
                return best_adv_prompt, rst_adv, best_loss_list, best_success_rate_list
            coordinate_grad = torch.zeros(max_control_length, word_length, dtype=torch.float32).to(self.device)
            for prompt in prompts:
                coordinate_grad_item = self.get_gradients(prompt)
                # BUG: Why the length of control tokens can vary?
                coordinate_grad[:coordinate_grad_item.shape[0]] += coordinate_grad_item
                del coordinate_grad_item
                gc.collect()
                torch.cuda.empty_cache()
            coordinate_grad = coordinate_grad / len(prompts)

            # Generate new candidates
            self.logger.debug(f"Iteration {i}: Generating new candidates ...")
            with torch.no_grad():
                # The candidate generation only depends on the original control string and the grad. Use the first valid prompt as input.
                example_prompt = prompts[0]
                new_controls = self.generate_candidates(example_prompt, coordinate_grad, return_prompts=False)
                self.logger.info(f"Iteration {i}: New candidates of adv control strings: {new_controls}")

            if self.vocab_check == "Strict":
                file_name = "data/toxic_words.txt"
                with open(file_name, "r") as f:
                    data = [line.strip() for line in f.readlines()]
                filtered_controls = [string for string in new_controls if not any(word in string for word in data)]
                if filtered_controls:
                    new_controls = filtered_controls
                self.logger.info(f"Iteration {i}: Filtered candidates of adv control strings: {new_controls}")
                
            # Select the best candidate
            self.logger.debug(f"Iteration {i}: Selecting the best candidate ...")
            losses = torch.zeros(len(new_controls), dtype=torch.float32)
            is_success_flags = []
            # NOTE: all_new_prompts is currently not used. It records all PromptSlices (num_user_prompts * num_candidates)
            all_new_prompts = []
            count = 0
            with torch.no_grad():
                for prompt, original_prompt in zip(prompts, original_prompts):
                    new_prompts = [self.prompt_manager.build(prompt.instruction, _control, prompt.target, prompt.insert_pos) for _control in new_controls]
                    all_new_prompts.append(new_prompts)
                    losses_item = self.loss_function(new_prompts, original_prompt)
                    if not torch.all(torch.isnan(losses_item)):
                        losses += losses_item
                    else:
                        count += 1
                    del losses_item
                    is_success_flags_item = self.check_success_batch(new_prompts, test_substr=[target_response])
                    is_success_flags.append(is_success_flags_item)
                    gc.collect()
                    torch.cuda.empty_cache()
            losses = losses / (len(prompts)-count)
            self.logger.info(f"Losses for each candidate: {losses}")
            success_rate = np.mean(np.asarray(is_success_flags), axis=0)
            self.logger.info(f"Success rate for each candidate: {success_rate}")

            # Candidates having the success rate below the threshold will be directly ignored.
            is_success_flags_after_thres = success_rate > self.candidate_config["universal_success_thres"]
            best_prompt_id = self.select_best_candidate(all_new_prompts, losses, is_success_flags_after_thres)
            prompts = [self.prompt_manager.build(original_prompt.instruction, new_controls[best_prompt_id], target_response, original_prompt.insert_pos) for original_prompt in original_prompts]
            best_adv_prompt, best_loss, best_success_rate, best_is_success_flag = new_controls[best_prompt_id], losses[best_prompt_id], success_rate[best_prompt_id], is_success_flags_after_thres[best_prompt_id]
            self.logger.info("Iteration {}: \n new_best_adv_prompt: {} \n new_best_adv_prompt_length : {} \n current loss: {} \n success rate: {} \n".format(i, best_adv_prompt,len(best_adv_prompt),best_loss, best_success_rate))

            if best_is_success_flag:
                fail_count = 0
                rst_adv_prompt = best_adv_prompt
                rst_adv.append(best_adv_prompt)
                best_loss_list.append(best_loss.item() if isinstance(best_loss, torch.Tensor) else best_loss)
                best_success_rate_list.append(best_success_rate)
            else:
                fail_count += 1
            if fail_count >= self.max_fails:
                break

            del coordinate_grad, losses
            gc.collect()
            torch.cuda.empty_cache()
        
        if best_loss_list:
            min_loss_index = best_loss_list.index(min(best_loss_list))  # Get the index of the minimum loss
            rst_adv_prompt = rst_adv[min_loss_index]  # Get the corresponding adv string from rst_adv
            self.logger.info(f"Selected adv_prompt with the lowest loss: {rst_adv_prompt}, with loss {best_loss_list[min_loss_index]}")
        else:
            self.logger.warning("No valid loss found. Keeping the default rst_adv_prompt.")

        return rst_adv_prompt, rst_adv, best_loss_list, best_success_rate_list

    def test_attack(self,
        user_prompt="an example instruction",
        adv_prompt="an example adversarial prompt",
        target_response="an example response",
    ):
        loss_list = []
        all_is_success = []
        flag = False
        if isinstance(adv_prompt, list):
            flag = True
            results = {}
        if not isinstance(adv_prompt, list):
            adv_prompt = [adv_prompt]
        if not isinstance(user_prompt, list):
            user_prompt = [user_prompt]
        for adv in adv_prompt:
            losses = []
            for user in user_prompt:
                prompt = self.prompt_manager.build(user, adv, target_response)
                input = prompt.input_ids.to(self.model.device).unsqueeze(0)
                logits = self.model.forward(input).logits
                logit = logits[:,prompt.loss_slice,:].to(self.model.device)
                label = input[:,prompt.target_slice].to(self.model.device)
                loss = self.loss_config["ce"] * F.cross_entropy(logit.view(-1, logit.size(-1)), label.view(-1)).detach()
                length_loss = torch.Tensor([len(adv)])
                length_loss = (self.loss_config["len"] * length_loss ** 2).to(self.model.device)
                total_loss = loss + length_loss
                losses.append(total_loss)
                gen_config = self.model.generation_config
                gen_config.max_new_tokens = 256
                del logit, label
                torch.cuda.empty_cache()
                # self.logger.info(f"Input:{self.tokenizer.decode(prompt.input_ids).strip()}")
                output = self.tokenizer.decode(self.generate(prompt)).strip()
                # self.logger.info(f"\nOutput: {output}")
                is_success = target_response in output
                all_is_success.append(is_success)
            mean_loss = torch.mean(torch.tensor(losses))
            if flag:
                results[adv] = mean_loss
            loss_list.append(mean_loss)

        if flag:
            sorted_results = dict(sorted(results.items(), key=lambda item: item[1]))
            return all_is_success, sorted_results
        else:
            return all_is_success, loss_list
        
    def test_attack_success(self,
        user_prompt="an example instruction",
        adv_prompt="an example adversarial prompt",
        target_response="an example response",
    ):
        if not isinstance(user_prompt, list):
            user_prompt = [user_prompt]
        prompts = []

        for instruction in user_prompt:
            prompt = self.prompt_manager.build(instruction, adv_prompt, target_response)
            prompts.append(prompt)

        success = self.check_success_batch(prompts, [target_response])
        return success
