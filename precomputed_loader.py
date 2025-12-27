# Save as precomputed_loader.py
import torch
from torch.utils.data import Dataset
import os

class PrecomputedVGDataset(Dataset):
    def __init__(self, data_dir="datasets/vg/precomputed_features"):
        self.data_dir = data_dir
        # List all .pt files
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        data = torch.load(path)
        return data