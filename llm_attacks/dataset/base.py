# This is only a placeholder

class BaseDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        BaseDataset.build(self)

    """Prepare everything. Read file content to memory.
    """
    def build(self):
        pass

    """Multiple ways to define dataset API
        (1) reload __len__ and __getitem__ similar to torch.Dataset
        (2) Simply define a function like `def get(self, idx)`
        (3) return a generator like `def generator(self): for i in range(len(data)) yield data[i]`
    """