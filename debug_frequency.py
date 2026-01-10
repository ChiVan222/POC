import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from collections import Counter
import os
from model import DecoupledSemanticSGG
from train import build_text_prototypes_cached, SPATIAL_CLASSES, ACTION_CLASSES, H5LazyDataset
from datasets.vg import VG150_PREDICATES

CHECKPOINT_PATH = "checkpoints/epoch_50.pth" 
TEST_H5 = "vg_data/features/test_features_negatives.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

s_map = {n: i for i, n in enumerate(SPATIAL_CLASSES)}
a_map = {n: i for i, n in enumerate(ACTION_CLASSES)}
vg_to_s = np.array([s_map.get(p, -100) for p in VG150_PREDICATES])
vg_to_a = np.array([a_map.get(p, -100) for p in VG150_PREDICATES])

@torch.no_grad()
def debug_top_predicates():
    model = DecoupledSemanticSGG(vis_dim=1024, geo_dim=8, embed_dim=768).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights from {CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    s_txt = build_text_prototypes_cached("|".join(SPATIAL_CLASSES), DINO_CKPT, DINO_CFG)
    a_txt = build_text_prototypes_cached("|".join(ACTION_CLASSES), DINO_CKPT, DINO_CFG)

    dataset = H5LazyDataset(TEST_H5)
    test_loader = DataLoader(dataset, batch_size=1, collate_fn=debug_collate, num_workers=0)

    model_preds_all, gt_labels_all = [], []
    confidences_all, margins_all = [], [] 
    
    for batch in tqdm(test_loader, desc="Calculating Metrics"):
        if batch is None: continue
        
        geo, vis = batch['geo'].to(DEVICE), batch['vis'].to(DEVICE).float()
        gt_ids = batch['pred'].cpu().numpy()

        s_logits, a_logits, *_ = model(geo, vis, s_txt, a_txt)
        
        s_probs = F.softmax(s_logits, dim=-1).cpu().numpy()
        a_probs = F.softmax(a_logits / 0.07, dim=-1).cpu().numpy()

        num_rels = geo.shape[0]
        unified_scores = np.zeros((num_rels, len(VG150_PREDICATES)))
        
        for vg_id in range(1, len(VG150_PREDICATES)):
            s_idx, a_idx = vg_to_s[vg_id], vg_to_a[vg_id]
            if s_idx != -100: 
                unified_scores[:, vg_id] = s_probs[:, s_idx]
            elif a_idx != -100: 
                unified_scores[:, vg_id] = a_probs[:, a_idx]

        sorted_scores = np.sort(unified_scores, axis=1)
        top1_val = sorted_scores[:, -1]
        top2_val = sorted_scores[:, -2]
        margins = top1_val - top2_val

        top_indices = np.argmax(unified_scores[:, 1:], axis=1) + 1
        
        model_preds_all.extend(top_indices.tolist())
        gt_labels_all.extend(gt_ids.tolist())
        confidences_all.extend(top1_val.tolist())
        margins_all.extend(margins.tolist())

    model_counts = Counter(model_preds_all)
    gt_counts = Counter(gt_labels_all)
    
    correct_counts = Counter()
    for p, g in zip(model_preds_all, gt_labels_all):
        if p == g:
            correct_counts[p] += 1
    
    pred_stats = {i: {"conf": [], "margin": []} for i in range(len(VG150_PREDICATES))}
    for p_id, conf, marg in zip(model_preds_all, confidences_all, margins_all):
        pred_stats[p_id]["conf"].append(conf)
        pred_stats[p_id]["margin"].append(marg)

    header = f"{'Predicate Name':<20} | {'Model':<8} | {'GT':<8} | {'Bias':<7} | {'Recall':<8} | {'Avg Conf':<9} | {'Avg Margin'}"
    print("\n" + "="*125)
    print(header)
    print("-" * 125)
    
    for pid, m_c in model_counts.most_common(35):
        p_name = VG150_PREDICATES[pid]
        g_c = gt_counts[pid]
        
        bias = f"{m_c / g_c:.1f}x" if g_c > 0 else "NEW"
        
        # Calculate Recall
        recall_val = (correct_counts[pid] / g_c) if g_c > 0 else 0.0
        recall_str = f"{recall_val:.2%}"
        
        avg_conf = np.mean(pred_stats[pid]["conf"])
        avg_marg = np.mean(pred_stats[pid]["margin"])
        
        flag = "[!] Weak Decision" if avg_marg < 0.05 else ""
        if recall_val < 0.05 and g_c > 50: flag = "[!] Low Recall"

        print(f"{p_name:<20} | {m_c:<8} | {g_c:<8} | {bias:<7} | {recall_str:<8} | {avg_conf:<9.4f} | {avg_marg:.4f} {flag}")
    
    print("="*125)

if __name__ == "__main__":
    debug_top_predicates()