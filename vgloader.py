import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import numpy as np

class VGDataset(Dataset):
    def __init__(self, split='train', 
                 img_dir="vg_data/VG_100K", 
                 dict_file="vg_data/stanford_filtered/VG-SGG-dicts-with-attri.json", 
                 roi_file="vg_data/stanford_filtered/VG-SGG-with-attri.h5"):
        
        self.img_dir = img_dir
        
        # 1. Load Dictionary
        if not os.path.exists(dict_file):
             raise FileNotFoundError(f"Dictionary not found: {dict_file}")
             
        with open(dict_file, 'r') as f:
            data = json.load(f)
            self.ind_to_classes = data['idx_to_label']
            self.ind_to_predicates = data['idx_to_predicate']
            
        # 2. Open HDF5
        if not os.path.exists(roi_file):
             raise FileNotFoundError(f"HDF5 not found: {roi_file}")
             
        self.roi_h5 = h5py.File(roi_file, 'r')
        
        # Handle Split
        split_flag = 0 if split == 'train' else 2
        if 'split' in self.roi_h5:
            self.split_mask = self.roi_h5['split'][:] == split_flag
            self.img_indices = np.where(self.split_mask)[0]
        else:
            self.img_indices = np.arange(self.roi_h5['labels'].shape[0])
        
        # Cache pointers
        self.img_to_first_box = self.roi_h5['img_to_first_box'][:]
        self.img_to_last_box = self.roi_h5['img_to_last_box'][:]
        self.img_to_first_rel = self.roi_h5['img_to_first_rel'][:]
        self.img_to_last_rel = self.roi_h5['img_to_last_rel'][:]
        
        # Load image mapping if available
        img_data_path = os.path.join(os.path.dirname(dict_file), "image_data.json")
        if os.path.exists(img_data_path):
            with open(img_data_path, 'r') as f:
                self.image_data = json.load(f)
        else:
            self.image_data = None

    def __len__(self):
        return len(self.img_indices)

    def __getitem__(self, idx):
        img_idx = self.img_indices[idx]
        
        # A. Get Image Path
        img_path = None
        if self.image_data:
             try:
                 img_info = self.image_data[img_idx]
                 if 'image_id' in img_info:
                    img_path = os.path.join(self.img_dir, f"{img_info['image_id']}.jpg")
             except IndexError:
                 pass
        
        if img_path is None:
             img_path = os.path.join(self.img_dir, "unknown.jpg") 

        # B. Get Boxes & Labels
        i_s = self.img_to_first_box[img_idx]
        i_e = self.img_to_last_box[img_idx]
        
        boxes = torch.from_numpy(self.roi_h5['boxes_1024'][i_s:i_e+1]).float()
        labels = torch.from_numpy(self.roi_h5['labels'][i_s:i_e+1]).long().squeeze()
        
        # C. Get Relations (The Fix!)
        rel_s = self.img_to_first_rel[img_idx]
        rel_e = self.img_to_last_rel[img_idx]
        
        if rel_s < 0 or rel_e < 0:
            relations = torch.zeros((0, 3)).long()
        else:
            # 1. Read [Subj, Obj]
            rel_indices = torch.from_numpy(self.roi_h5['relationships'][rel_s:rel_e+1]).long()
            
            # 2. Read [Predicate] (Labels)
            if 'predicates' in self.roi_h5:
                predicates = torch.from_numpy(self.roi_h5['predicates'][rel_s:rel_e+1]).long()
                # Ensure shape is [N, 1] so we can concatenate
                if predicates.dim() == 1:
                    predicates = predicates.unsqueeze(1)
                
                # 3. Merge into [Subj, Obj, Pred] -> [N, 3]
                relations = torch.cat((rel_indices, predicates), dim=1)
            else:
                # Fallback if predicates are somehow inside relationships (rare for this file type)
                relations = rel_indices

            # Normalize global box indices to local image indices (0..N)
            relations[:, 0] = relations[:, 0] - i_s
            relations[:, 1] = relations[:, 1] - i_s

        return {
            'img_path': img_path,
            'boxes': boxes,          # [N, 4]
            'labels': labels,        # [N]
            'relations': relations,  # [R, 3]
            'img_idx': img_idx
        }
        
    def get_text_label(self, label_id):
        return self.ind_to_classes.get(str(label_id), "unknown")
        
    def get_text_predicate(self, pred_id):
        return self.ind_to_predicates.get(str(pred_id), "unknown")