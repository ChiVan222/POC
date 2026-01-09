import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from collections import Counter
import os

# --- PROJECT IMPORTS ---
from model_decoupled import DecoupledSemanticSGG
from train import build_text_prototypes_cached, SPATIAL_CLASSES, ACTION_CLASSES, H5LazyDataset
from datasets.vg import VG150_PREDICATES

# ============================================================
# 1. CONFIGURATION
# ============================================================
CHECKPOINT_PATH = "checkpoints/epoch_50.pth" 
TEST_H5 = r"C:\Users\van\Desktop\SGG_data\test_features_negatives.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Explicit paths to ensure BERT/SwinB sync
DINO_CKPT = "weights/vg-pretrain-coco-swinb.pth"
DINO_CFG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_ovr.py"

def debug_collate(batch):
    batch = [b for b in batch if b is not None and 'pred' in b]
    if not batch: return None
    return {
        'geo': torch.from_numpy(np.concatenate([b['geo'] for b in batch], 0)),
        'vis': torch.from_numpy(np.concatenate([b['vis'] for b in batch], 0)),
        'pred': torch.from_numpy(np.concatenate([b['pred'] for b in batch], 0))
    }

# Unified Mapping Logic
s_map = {n: i for i, n in enumerate(SPATIAL_CLASSES)}
a_map = {n: i for i, n in enumerate(ACTION_CLASSES)}
vg_to_s = np.array([s_map.get(p, -100) for p in VG150_PREDICATES])
vg_to_a = np.array([a_map.get(p, -100) for p in VG150_PREDICATES])

@torch.no_grad()
def debug_top_predicates():
    model = DecoupledSemanticSGG(vis_dim=1024, geo_dim=8, embed_dim=768).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # Pass explicit paths to the cached builder
    s_txt = build_text_prototypes_cached("|".join(SPATIAL_CLASSES), DINO_CKPT, DINO_CFG)
    a_txt = build_text_prototypes_cached("|".join(ACTION_CLASSES), DINO_CKPT, DINO_CFG)

    test_loader = DataLoader(H5LazyDataset(TEST_H5), batch_size=1, collate_fn=debug_collate, num_workers=0)

    model_preds_all = []
    gt_labels_all = []

    for batch in tqdm(test_loader, desc="Scanning Test Set"):
        if batch is None: continue
        
        geo, vis = batch['geo'].to(DEVICE), batch['vis'].to(DEVICE).float()
        gt_ids = batch['pred'].cpu().numpy()

        s_logits, a_logits, *_ = model(geo, vis, s_txt, a_txt)
        
        # Scaling Action head to match training loss distribution
        s_probs = F.softmax(s_logits, dim=-1).cpu().numpy()
        a_probs = F.softmax(a_logits / 0.07, dim=-1).cpu().numpy()

        num_rels = geo.shape[0]
        unified_scores = np.zeros((num_rels, len(VG150_PREDICATES)))
        for vg_id in range(1, len(VG150_PREDICATES)):
            s_idx, a_idx = vg_to_s[vg_id], vg_to_a[vg_id]
            if s_idx != -100: unified_scores[:, vg_id] = s_probs[:, s_idx]
            elif a_idx != -100: unified_scores[:, vg_id] = a_probs[:, a_idx]

        # GET TOP NON-BACKGROUND PREDICTION
        # We find the argmax across the unified 51 predicates (skipping index 0)
        top_preds = np.argmax(unified_scores[:, 1:], axis=1) + 1
        
        model_preds_all.extend(top_preds.tolist())
        gt_labels_all.extend(gt_ids.tolist())

    model_counts = Counter(model_preds_all)
    gt_counts = Counter(gt_labels_all)

    print("\n" + "="*85)
    print(f"{'Predicate Name':<20} | {'Model Count':<15} | {'GT Count':<15} | {'Bias Ratio'}")
    print("-" * 85)
    
    # Sort by Model Count to see what the model is "obsessed" with
    for pid, m_c in model_counts.most_common(25):
        p_name = VG150_PREDICATES[pid]
        g_c = gt_counts[pid]
        # Bias Ratio: >1.0 means model over-predicts this compared to GT
        ratio = m_c / g_c if g_c > 0 else float('inf')
        ratio_str = f"{ratio:.2f}x" if g_c > 0 else "NEW"
        print(f"{p_name:<20} | {m_c:<15} | {g_c:<15} | {ratio_str}")
    print("="*85)

if __name__ == "__main__":
    debug_top_predicates()