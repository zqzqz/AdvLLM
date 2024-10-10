import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseLLM
from .chat_template import ChatTemplate


class LlamaGuardChatTemplate(ChatTemplate):
    def __init__(self, name, tokenizer):
        super().__init__(name)
        self.tokenizer = tokenizer

    def apply_chat_template(self, chat):
        return self.tokenizer.apply_chat_template(chat, tokenize=False)


class LlamaGuard(BaseLLM):
    def __init__(self, model_id="meta-llama/Meta-Llama-Guard-2-8B", tokenizer_id=None, model_config={}, tokenizer_config={}):
        super().__init__()
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id if tokenizer_id is not None else model_id
        self.model_config = {
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "ignore_mismatched_sizes": True,
        }
        self.model_config.update(model_config)
        self.tokenizer_config = {
            "trust_remote_code": True,
            "use_fast": False,
            "padding_side": 'left',
        }
        self.tokenizer_config.update(tokenizer_config)
        self.chat_template = None
        LlamaGuard.build(self)
    
    def build(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **self.model_config

        ).to(self.device).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_id,
            **self.tokenizer_config
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = 'left'

        # TODO: Can we remove this fastchat dependency?
        if "Llama-Guard-2" in self.model_id:
            template_name = "llama-guard-2"
        elif "Llama-Guard-3" in self.model_id:
            template_name = "llama-guard-3"
        else:
            template_name = "llama-guard"

        self.chat_template = LlamaGuardChatTemplate(template_name, self.tokenizer)

    def generate_token_ids(self, input_ids):
        input_ids = input_ids.to(self.model.device).unsqueeze(0)
        attn_masks = torch.ones_like(input_ids).to(self.model.device)
        output_ids = self.model.generate(input_ids, 
                                         attention_mask=attn_masks, 
                                         generation_config=self.model.generation_config,
                                         pad_token_id=self.tokenizer.pad_token_id)[0]
        return output_ids
    
    def get_attention_map(self, text, save=None):
        self.model.config.output_attentions = True
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        outputs = self.model(**inputs)
        attention_maps = outputs.attentions
        attention_maps = attention_maps[0][0].detach().cpu().numpy().sum(axis=0)
        sliced_attention_maps = attention_maps.copy()
        sliced_attention_maps[sliced_attention_maps > 1] = 1
        # print(sliced_attention_maps)

        if save is not None:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(30, 30))
            sns.heatmap(sliced_attention_maps, cmap='viridis')
            plt.savefig(save)
        self.model.config.output_attentions = False

        return attention_maps
