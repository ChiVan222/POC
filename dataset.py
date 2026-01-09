import torch
from torch.utils.data import Dataset
import os
import random
import json
import h5py
import numpy as np

# --- CONFIG ---
SPATIAL_CLASSES = [
    "above", "across", "against", "along", "at", "behind", "between", 
    "in", "in front of", "near", "next to", "on", "on back of", 
    "over", "under", "with"
]

ACTION_CLASSES = [
    "carrying", "eating", "flying in", "holding", "looking at", "playing", 
    "riding", "sitting on", "standing on", "using", "walking in", "walking on", 
    "watching", "wearing", "wears", "parked on", "hanging from", "growing on", 
    "covered in", "covering", "mounted on", "attached to", "painted on",
    "laying", "laying on", "leaning", "leaning on", "sitting", "standing",
    "and", "belonging to", "for", "from", "has", "made of", "of", 
    "part of", "says", "to"
]

class CachedDecoupledDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        # Open once to cache the number of images
        with h5py.File(self.h5_path, 'r') as f:
            self.num_images = f.attrs['num_images']

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # Open H5 in each worker to avoid thread-safety issues
        with h5py.File(self.h5_path, 'r') as f:
            str_idx = str(idx)
            if str_idx not in f:
                return None
            
            grp = f[str_idx]
            if grp.attrs.get('num_rel', 0) == 0:
                return None

            return {
                'geo': torch.from_numpy(grp['geo'][:]).float(),
                'vis': torch.from_numpy(grp['vis'][:]).float(), # Cast float16 to float32
                's_label': torch.from_numpy(grp['s_label'][:]).long(),
                'a_label': torch.from_numpy(grp['a_label'][:]).long()
            }

def collate_fn(batch):
    """Filters out None samples where images had no relationships."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    return torch.utils.data.dataloader.default_collate(batch)