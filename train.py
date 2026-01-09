import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import os
import h5py
import warnings
import gc
from functools import lru_cache

# --- PROJECT IMPORTS ---
from model_decoupled import DecoupledSemanticSGG
from GroundingDINO.groundingdino.util.inference import load_model

# ============================================================
# 1. CONFIGURATION
# ============================================================
warnings.filterwarnings('ignore')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

CONFIG = {
    "train_batch_size": 128,      # Optimized for SSD streaming
    "learning_rate": 1e-4,
    "num_epochs": 50,
    "warmup_epochs": 5,
    "save_frequency": 5,
    "bg_weight": 0.1,             # Learn background without letting it dominate
    "num_workers": 14,            # Physical cores on your Xeon CPU
}

TRAIN_H5 = r"C:\Users\van\Desktop\SGG_data\train_features_negatives.h5"

# NEW CLASS LISTS (Including __background__ at Index 0)
SPATIAL_CLASSES = ["__background__", "above", "across", "against", "along", "at", "behind", "between", "in", "in front of", "near", "on", "on back of", "over", "under", "with"]
ACTION_CLASSES = ["__background__", "carrying", "eating", "flying in", "holding", "laying on", "looking at", "lying on", "playing", "riding", "sitting on", "standing on", "using", "walking in", "walking on", "watching", "wearing", "wears"]

# ============================================================
# 2. DATASET (On-the-fly Background Mapping)
# ============================================================

class H5LazyDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, 'r') as f:
            self.keys = list(f.keys())

    def __len__(self): return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            grp = f[self.keys[idx]]
            
            # Use .astype if you need to ensure consistent types
            return {
                'geo': torch.from_numpy(grp['geo'][:]).float(),
                'vis': torch.from_numpy(grp['vis'][:]).float(),
                's_label': torch.from_numpy(grp['s_label'][:]).long(),
                'a_label': torch.from_numpy(grp['a_label'][:]).long(),
                'pred': torch.from_numpy(grp['pred'][:]).long()  # <--- CRITICAL ADDITION
            }
def train_collate(batch):
    batch = [b for b in batch if b is not None]
    return {
        'geo': torch.cat([b['geo'] for b in batch], 0),
        'vis': torch.cat([b['vis'] for b in batch], 0),
        's_label': torch.cat([b['s_label'] for b in batch], 0),
        'a_label': torch.cat([b['a_label'] for b in batch], 0),
    }

# ============================================================
# 3. BALANCED LOSS FUNCTIONS
# ============================================================

class WeightedDecoupledLoss(nn.Module):
    def __init__(self, num_classes, reg_w=0.1, is_action=False):
        super().__init__()
        self.reg_w = reg_w
        self.is_action = is_action
        
        # Weighted CE: 0.1 for background, 1.0 for relations
        weights = torch.ones(num_classes).to(DEVICE)
        weights[0] = CONFIG["bg_weight"]
        self.ce = nn.CrossEntropyLoss(weight=weights)
        
    def forward(self, logits, labels, feat, text_embeds):
        # Apply temperature if it's the action head
        target_logits = logits / 0.07 if self.is_action else logits
        
        loss_cls = self.ce(target_logits, labels)
        
        # Semantic alignment for non-background samples only
        mask = labels > 0
        if self.reg_w > 0 and mask.any():
            loss_reg = 1.0 - F.cosine_similarity(F.normalize(feat[mask], dim=-1), 
                                                 text_embeds[labels[mask]], dim=-1).mean()
            return loss_cls + self.reg_w * loss_reg
        return loss_cls

# ============================================================
# 4. MAIN TRAINING EXECUTION
# ============================================================

@lru_cache(maxsize=2)
def build_text_prototypes_cached(classes_key, model_path, config_path):
    print(f">>> Encoding Text Prototypes...")
    dino = load_model(config_path, model_path).to(DEVICE)
    class_names = classes_key.split('|')
    all_embeds = []
    for i in range(0, len(class_names), 32):
        batch = class_names[i:i+32]
        tokens = dino.tokenizer(batch, padding="max_length", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = dino.bert(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask)
            all_embeds.append(F.normalize(out.last_hidden_state[:, 0], dim=-1).cpu())
    del dino
    torch.cuda.empty_cache()
    return torch.cat(all_embeds).to(DEVICE)

def main():
    s_txt = build_text_prototypes_cached("|".join(SPATIAL_CLASSES), "weights/vg-pretrain-coco-swinb.pth", "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_ovr.py")
    a_txt = build_text_prototypes_cached("|".join(ACTION_CLASSES), "weights/vg-pretrain-coco-swinb.pth", "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_ovr.py")

    train_loader = DataLoader(H5LazyDataset(TRAIN_H5), batch_size=CONFIG["train_batch_size"], 
                              shuffle=True, collate_fn=train_collate, num_workers=CONFIG["num_workers"], 
                              persistent_workers=True)

    # Heads: 16 (15+1) and 18 (17+1)
    model = DecoupledSemanticSGG(vis_dim=1024, geo_dim=8, embed_dim=768).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    scaler = torch.cuda.amp.GradScaler()
    
    s_crit = WeightedDecoupledLoss(len(SPATIAL_CLASSES), reg_w=0.1, is_action=False)
    a_crit = WeightedDecoupledLoss(len(ACTION_CLASSES), reg_w=0.2, is_action=True)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            geo, vis = batch['geo'].to(DEVICE), batch['vis'].to(DEVICE)
            sl, al = batch['s_label'].to(DEVICE), batch['a_label'].to(DEVICE)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                sl_logits, al_logits, _, f_ali, _, _ = model(geo, vis, s_txt, a_txt)
                _, s_feat = model.spatial(geo, s_txt)
                loss = s_crit(sl_logits, sl, s_feat, s_txt) + 1.5 * a_crit(al_logits, al, f_ali, a_txt)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if (epoch + 1) % CONFIG["save_frequency"] == 0:
            torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pth")
        
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()