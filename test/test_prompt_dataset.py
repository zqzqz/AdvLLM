from test_base import *

from llm_attacks.dataset.prompt_dos_dataset import PromptDosDataset


def test_prompt_dos_dataset():
    dataset = PromptDosDataset(data_dir=os.path.join(workspace_root, "data"))
    print(dataset[0])


if __name__ == "__main__":
    test_prompt_dos_dataset()