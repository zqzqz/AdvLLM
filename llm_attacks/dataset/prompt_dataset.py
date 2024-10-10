import os
import json

from .base import BaseDataset


class PromptDataset(BaseDataset):
    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.data = {}
        PromptDataset.build(self)
    
    def build(self):
        self.data = {}
        for safety_category in ["safe", "unsafe"]:
            self.data[safety_category] = {}
            
            filenames = [filename for filename in os.listdir(os.path.join(self.data_dir, safety_category)) if filename.endswith('.jsonl')]
            for filename in filenames:
                task_category = filename[:-6]
                self.data[safety_category][task_category] = []

                with open(os.path.join(self.data_dir, safety_category, filename), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        prompt = json.loads(line).get(f"{safety_category}_string")
                        self.data[safety_category][task_category].append(prompt)
