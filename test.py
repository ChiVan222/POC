import torch
import torch.nn.functional as F
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
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

# Official VG Paths
IMG_DIR = r"D:\SceneGraphGeneration\POC\vg_data\VG_100K"
ROIDB_FILE = r"D:\SceneGraphGeneration\POC\vg_data\stanford_filtered\VG-SGG.h5"
DICT_FILE = r"D:\SceneGraphGeneration\POC\vg_data\stanford_filtered\VG-SGG-dicts.json"
IMAGE_FILE = r"D:\SceneGraphGeneration\POC\vg_data\stanford_filtered\image_data.json"

# ============================================================
# 2. VISUALIZATION ENGINE
# ============================================================
def draw_full_sgg(image_path, gt_triplets, model_preds, boxes):
    """Visualizes detections and compares Top-K predictions to GT"""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Draw boxes with IDs for reference
    for i, box in enumerate(boxes):
        draw.rectangle([box[0], box[1], box[2], box[3]], outline="red", width=2)
        draw.text((box[0], box[1]), f"o{i}", fill="yellow")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), gridspec_kw={'width_ratios': [1, 1.2]})
    ax1.imshow(img)
    ax1.set_title(f"Image Source: {os.path.basename(image_path)}")
    ax1.axis('off')

    ax2.axis('off')
    header = f"{'Rank':<5} | {'Model Prediction':<35} | {'Status':<15} | {'Score':<6}\n"
    header += "-" * 80 + "\n"
    
    y_pos = 0.95
    ax2.text(0.01, y_pos, header, transform=ax2.transAxes, fontsize=9, family='monospace', fontweight='bold')
    y_pos -= 0.04

    # Iterate through Top-K relation predictions
    for i, res in enumerate(model_preds[:50]):
        triplet_str = f"(o{res['s']})--{res['p']}-->(o{res['o']})"
        
        is_correct = False
        gt_label = "None"
        # Match using triplet identity (Subject ID, Object ID)
        for gt_s, gt_p, gt_o in gt_triplets:
            if res['s'] == gt_s and res['o'] == gt_o:
                gt_label = gt_p
                if res['p'] == gt_p: 
                    is_correct = True
                break
        
        color = 'green' if is_correct else 'black'
        status = "✅ MATCH" if is_correct else (f"❌(GT:{gt_label})" if gt_label != "None" else "🆕 DISCOVERY")
        
        line = f"{i+1:<5} | {triplet_str:<35} | {status:<15} | {res['score']:.4f}"
        ax2.text(0.01, y_pos, line, transform=ax2.transAxes, fontsize=8, family='monospace', color=color)
        y_pos -= 0.018
        if y_pos < 0.05: break

    plt.tight_layout()
    plt.show()

# ============================================================
# 3. INFERENCE LOOP
# ============================================================
def run_inference_viz(num_samples=5):
    # Load Model & Weights
    model = DecoupledSemanticSGG(vis_dim=1024, geo_dim=8, embed_dim=768).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # Load Text Prototypes
    s_txt = build_text_prototypes_cached("|".join(SPATIAL_CLASSES), "weights/vg-pretrain-coco-swinb.pth", "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_ovr.py")
    a_txt = build_text_prototypes_cached("|".join(ACTION_CLASSES), "weights/vg-pretrain-coco-swinb.pth", "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_ovr.py")

    # Initialize Dataset
    vg_base = VGDataset(split="test", img_dir=IMG_DIR, roidb_file=ROIDB_FILE, 
                        dict_file=DICT_FILE, image_file=IMAGE_FILE, num_val_im=0)
    
    # Access H5 directly for Identity Alignment
    h5_f = h5py.File(TEST_H5, 'r')
    available_indices = list(h5_f.keys())
    selected_indices = np.random.choice(available_indices, num_samples, replace=False)

    for idx_str in selected_indices:
        idx = int(idx_str)
        grp = h5_f[idx_str]
        
        # Identity-Aware Feature Loading
        geo = torch.from_numpy(grp['geo'][:]).to(DEVICE).float()
        vis = torch.from_numpy(grp['vis'][:]).to(DEVICE).float()
        h5_sub = grp['sub_id'][:]
        h5_obj = grp['obj_id'][:]

        with torch.no_grad():
            s_logits, a_logits, *_ = model(geo, vis, s_txt, a_txt)
            # Use Temperature Scaling for Action head calibration
            s_probs = F.softmax(s_logits, dim=-1).cpu().numpy()
            a_probs = F.softmax(a_logits / 0.07, dim=-1).cpu().numpy()

        model_preds = []
        for i in range(geo.shape[0]):
            s_score, s_id = np.max(s_probs[i]), np.argmax(s_probs[i])
            a_score, a_id = np.max(a_probs[i]), np.argmax(a_probs[i])
            
            # Select most confident prediction per head
            if s_score > a_score:
                pred_name, final_score = SPATIAL_CLASSES[s_id], s_score
            else:
                pred_name, final_score = ACTION_CLASSES[a_id], a_score

            model_preds.append({
                's': int(h5_sub[i]), 
                'o': int(h5_obj[i]), 
                'p': pred_name, 
                'score': float(final_score)
            })

        # Global Triplet Ranking for visualization
        model_preds.sort(key=lambda x: x['score'], reverse=True)
        
        # Get Official GT and Image Path
        # Fix: Access file_name via the images list dictionary
        img_metadata = vg_base.images[idx] 
        img_filename = img_metadata['file_name']
        img_full_path = os.path.join(IMG_DIR, os.path.basename(img_filename))
        
        _, raw_target = vg_base[idx]
        boxes = raw_target['boxes'].numpy()
        gt_list = []
        for s, o, p_id in raw_target['edges'].numpy():
            gt_list.append((int(s), VG150_PREDICATES[p_id], int(o)))

        draw_full_sgg(img_full_path, gt_list, model_preds, boxes)

    h5_f.close()

if __name__ == "__main__":
    run_inference_viz(num_samples=3)