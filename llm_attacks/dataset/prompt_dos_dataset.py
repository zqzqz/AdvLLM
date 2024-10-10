import copy
import random

from .prompt_dataset import PromptDataset
from sklearn.model_selection import train_test_split

class PromptDosDataset(PromptDataset):
    def __init__(self, data_dir, num_cases=100, num_universal=30, test_cases=100, safe_task_category=None, unsafe_task_category="new_filter_unsafe_data", task=None, test_size=0.2):
        super().__init__(data_dir)
        self.num_cases = num_cases
        self.num_universal = num_universal
        self.test_cases = test_cases
        self.safe_task_category = safe_task_category
        self.unsafe_task_category = unsafe_task_category
        self.train_data = []
        self.test_data = []
        self.task = task
        self.test_size = test_size
        PromptDosDataset.build(self)

    def build(self):
        random.seed(42)
        total_unsafe_prompts = copy.deepcopy(self.data["unsafe"][self.unsafe_task_category])
        if self.safe_task_category is not None:
            if isinstance(self.safe_task_category, list):
                total_safe_prompts = []
                for category in self.safe_task_category:
                    total_safe_prompts.extend(copy.deepcopy(self.data["safe"][category]))
            elif isinstance(self.safe_task_category, str):
                total_safe_prompts = copy.deepcopy(self.data["safe"][self.safe_task_category])
            else:
                raise ValueError("safe_task_category must be a list or a string")
        else:
            total_safe_prompts = sum(list(self.data["safe"].values()), [])
        train_safe_prompts, test_safe_prompts = train_test_split(
            total_safe_prompts, test_size=self.test_size, random_state=42,
        )
        if self.task == "filter":
            total_unsafe_prompts = random.sample(total_unsafe_prompts, k=self.num_cases)
            train_safe_prompts = random.sample(train_safe_prompts, self.num_universal)
            for case_id in range(len(total_unsafe_prompts)):
                self.train_data.append({
                    "safe_prompt": train_safe_prompts,
                    "unsafe_prompt":total_unsafe_prompts[case_id],
                })
        elif self.task == "task_universal":
            test_safe_prompts = test_safe_prompts[:min(self.test_cases, len(test_safe_prompts))]
            self.test_data.append({
                "safe_prompt": test_safe_prompts,
            })
            train_safe_prompts = random.choices(train_safe_prompts, k=self.num_universal*self.num_cases)
            for i in range(self.num_cases):
                safe_prompts = train_safe_prompts[i*self.num_universal:(i+1)*self.num_universal]
                self.train_data.append({
                    "safe_prompt": safe_prompts,
                    "unsafe_prompt":total_unsafe_prompts,
                })
        elif self.task == "single":
            train_safe_prompts = random.sample(total_safe_prompts, k=self.num_cases)
            for case_id in range(self.num_cases):
                self.train_data.append({
                    "safe_prompt": train_safe_prompts[case_id],
                    "unsafe_prompt":total_unsafe_prompts,
                })
        elif self.task == "universal":
            test_safe_prompts = test_safe_prompts[:min(self.test_cases, len(test_safe_prompts))]
            self.test_data.append({
                "safe_prompt": test_safe_prompts,
            })
            train_safe_prompts = random.choices(train_safe_prompts, k=self.num_universal*self.num_cases)
            for i in range(self.num_cases):
                safe_prompts = train_safe_prompts[i*self.num_universal:(i+1)*self.num_universal]
                self.train_data.append({
                    "safe_prompt": safe_prompts,
                    "unsafe_prompt":total_unsafe_prompts,
                })
        else:
            raise(NotImplementedError)
    
    def __len__(self):
        return len(self.case_data)

    def __getitem__(self, idx):
        return self.case_data[idx]
