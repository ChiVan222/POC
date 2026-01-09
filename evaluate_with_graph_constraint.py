import torch
import torch.nn.functional as F
import h5py
import numpy as np
from tqdm import tqdm
import os

# --- PROJECT IMPORTS ---
from model_decoupled import DecoupledSemanticSGG
from train import build_text_prototypes_cached, SPATIAL_CLASSES, ACTION_CLASSES
from datasets.vg import VGDataset, VG150_PREDICATES 

# ============================================================
# 1. CONFIGURATION
# ============================================================
CHECKPOINT_PATH = "checkpoints/epoch_50.pth"
TEST_H5 = r"C:\Users\van\Desktop\SGG_data\test_features_with_ids.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_DIR = r"D:\SceneGraphGeneration\POC\vg_data\VG_100K"
ROIDB_FILE = r"D:\SceneGraphGeneration\POC\vg_data\stanford_filtered\VG-SGG.h5"
DICT_FILE = r"D:\SceneGraphGeneration\POC\vg_data\stanford_filtered\VG-SGG-dicts.json"
IMAGE_FILE = r"D:\SceneGraphGeneration\POC\vg_data\stanford_filtered\image_data.json"

K_VALS = [20, 50, 100]

# Category Split for Deep Analysis
PREDICATE_MAPPING = {
    "Spatial": ["above", "across", "against", "along", "at", "behind", "between", 
                "in", "in front of", "near", "on", "on back of", "over", "under", "with"],
    "Action": ["carrying", "eating", "flying in", "holding", "laying on", "looking at", 
               "lying on", "playing", "riding", "sitting on", "standing on", "using", 
               "walking in", "walking on", "watching", "wearing", "wears"],
    "Structural/Misc": ["and", "attached to", "belonging to", "covered in", "covering", "for", 
                        "from", "growing on", "hanging from", "has", "made of", "mounted on", 
                        "of", "painted on", "parked on", "part of", "says", "to"]
}

ID_TO_CATEGORY = {VG150_PREDICATES.index(p): cat for cat, plist in PREDICATE_MAPPING.items() 
                  for p in plist if p in VG150_PREDICATES}

s_map = {name: i for i, name in enumerate(SPATIAL_CLASSES)}
a_map = {name: i for i, name in enumerate(ACTION_CLASSES)}
vg_to_s = np.array([s_map.get(p, -1) for p in VG150_PREDICATES])
vg_to_a = np.array([a_map.get(p, -1) for p in VG150_PREDICATES])

@torch.no_grad()
def evaluate_official_full_analysis():
    print(f"\n🚀 Evaluating OFFICIAL Constrained PredCls + Category Analysis on {DEVICE}")
    
    model = DecoupledSemanticSGG(vis_dim=1024, geo_dim=8, embed_dim=768).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    s_txt = build_text_prototypes_cached("|".join(SPATIAL_CLASSES), "weights/vg-pretrain-coco-swinb.pth", "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_ovr.py")
    a_txt = build_text_prototypes_cached("|".join(ACTION_CLASSES), "weights/vg-pretrain-coco-swinb.pth", "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_ovr.py")

    vg_base = VGDataset(split="test", img_dir=IMG_DIR, roidb_file=ROIDB_FILE, dict_file=DICT_FILE, image_file=IMAGE_FILE, num_val_im=0)
    h5_f = h5py.File(TEST_H5, 'r')

    # Analysis Accumulators
    cat_hits = {cat: {k: 0 for k in K_VALS} for cat in PREDICATE_MAPPING}
    cat_totals = {cat: 0 for cat in PREDICATE_MAPPING}
    per_pred_hits = {p: {k: 0 for k in K_VALS} for p in range(1, len(VG150_PREDICATES))}
    per_pred_totals = {p: 0 for p in range(1, len(VG150_PREDICATES))}

    for idx in tqdm(range(len(vg_base)), desc="Evaluating"):
        str_idx = str(idx)
        if str_idx not in h5_f: continue
        grp = h5_f[str_idx]
        
        # 1. Inference & Unified Scores
        geo = torch.from_numpy(grp["geo"][:]).to(DEVICE).float()
        vis = torch.from_numpy(grp["vis"][:]).to(DEVICE).float()
        s_logits, a_logits, *_ = model(geo, vis, s_txt, a_txt)
        s_probs = F.softmax(s_logits, dim=-1).cpu().numpy()
        a_probs = F.softmax(a_logits / 0.07, dim=-1).cpu().numpy() #

        num_pairs = geo.shape[0]
        unified = np.zeros((num_pairs, len(VG150_PREDICATES)))
        for p in range(1, len(VG150_PREDICATES)):
            if vg_to_s[p] != -1: unified[:, p] = s_probs[:, vg_to_s[p]]
            elif vg_to_a[p] != -1: unified[:, p] = a_probs[:, vg_to_a[p]]

        # 2. Graph Constraint (Argmax per pair)
        pair_best_pred = np.argmax(unified[:, 1:], axis=1) + 1
        pair_best_score = np.max(unified[:, 1:], axis=1)
        
        # 3. Relation Preparation
        h5_sub, h5_obj = grp["sub_id"][:], grp["obj_id"][:]
        pred_relations = []
        for i in range(num_pairs):
            pred_relations.append((int(h5_sub[i]), int(pair_best_pred[i]), int(h5_obj[i])))
        
        ranking = np.argsort(pair_best_score)[::-1]

        # 4. Matching & Analysis
        _, raw_target = vg_base[idx]
        gt_edges = raw_target["edges"].numpy()
        gt_relations = set((int(s), int(p), int(o)) for s, o, p in gt_edges)
        
        # Update category totals based on GT
        for s, p, o in gt_relations:
            cat = ID_TO_CATEGORY.get(p, "Structural/Misc")
            cat_totals[cat] += 1
            per_pred_totals[p] += 1

        # Calculate hits for each K
        for k in K_VALS:
            for r_idx in ranking[:k]:
                rel = pred_relations[r_idx]
                if rel in gt_relations:
                    p_id = rel[1]
                    cat = ID_TO_CATEGORY.get(p_id, "Structural/Misc")
                    cat_hits[cat][k] += 1
                    per_pred_hits[p_id][k] += 1

    # --- 5. REPORT GENERATION ---
    print("\n" + "=" * 90)
    print(f"{'CATEGORY':<20} | {'R@20':<10} | {'R@50':<10} | {'R@100':<10} | {'COUNT'}")
    print("-" * 90)

    for cat in PREDICATE_MAPPING:
        row = [f"{cat:<20}"]
        for k in K_VALS:
            val = cat_hits[cat][k] / cat_totals[cat] if cat_totals[cat] else 0
            row.append(f"{val:<10.4f}")
        row.append(f"{cat_totals[cat]}")
        print(" | ".join(row))

    print("-" * 90)

    # Mean Recall (mR@K) logic
    mr_row = [f"{'MEAN RECALL (mR@K)':<20}"]
    for k in K_VALS:
        recalls = [per_pred_hits[p][k] / per_pred_totals[p] for p in range(1, 51) if per_pred_totals[p] > 0]
        mr_row.append(f"{np.mean(recalls):<10.4f}" if recalls else f"{0:<10.4f}")
    mr_row.append("-")
    print(" | ".join(mr_row))

    # Total Official Recall
    total_gt = sum(cat_totals.values())
    total_row = [f"{'TOTAL RECALL (R@K)':<20}"]
    for k in K_VALS:
        hits = sum(cat_hits[c][k] for c in cat_hits)
        total_row.append(f"{hits / total_gt:<10.4f}" if total_gt else f"{0:<10.4f}")
    total_row.append(f"{total_gt}")
    print(" | ".join(total_row))
    print("=" * 90)
    h5_f.close()

if __name__ == "__main__":
    evaluate_official_full_analysis()