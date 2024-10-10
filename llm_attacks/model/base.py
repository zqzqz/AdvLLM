import torch


class BaseLLM:
    def __init__(self, device="cuda:0"):
        self.model = None
        self.tokenizer = None
        self.conv_template = None
        self.device = device
        BaseLLM.build(self)

    def build(self):
        # Load everything. The model, the tokenizer, the conversation
        # template, and any configuration.
        pass
