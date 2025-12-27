import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image

# Import your modules
from pipeline import FeatureExtractor, SAMWrapper, ImageObject
from vgloader import VGDataset 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "datasets/vg/precomputed_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess():
    print(f"Initializing Pre-computation on {DEVICE}...")

    # 1. Init Vision Models
    # Using ViT-B for speed (make sure you downloaded the weights!)
    feat_extractor = FeatureExtractor(device=DEVICE) 
    sam_wrapper = SAMWrapper(SAM_TYPE="vit_b", SAM_CKPT="GroundingDINO/weights/sam_vit_b_01ec64.pth")

    # 2. Setup Dataset
    dataset = VGDataset(split='train')
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    print(f"Starting extraction... (Capping at 7,000 images)")

    for i, sample in tqdm(enumerate(loader),total =len(loader)): # Updated total for progress bar
        

        # ---------------

        # Resume capability
        save_path = os.path.join(OUTPUT_DIR, f"{i}.pt")
        if os.path.exists(save_path):
            continue

        if 'img_path' not in sample or not os.path.exists(sample['img_path']):
            continue

        img_path = sample['img_path']
        boxes = sample['boxes'].to(DEVICE)
        labels = sample['labels']
        relations = sample['relations']
        
        if len(relations) == 0: continue

        # --- A. NODE FEATURES (SAM) ---
        # Load image
        image_bgr = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        H, W, _ = image_rgb.shape
        
        # Fix 1024 Scaling
        scale_factor = max(H, W) / 1024.0
        boxes = boxes * scale_factor

        # Normalize for SAM
        boxes_norm = boxes.clone()
        boxes_norm[:, [0, 2]] /= W
        boxes_norm[:, [1, 3]] /= H
        
        # SAM Process
        with torch.no_grad():
            # Use autocast for speed
            with torch.amp.autocast('cuda'):
                masks, _ = sam_wrapper.process(img_path, boxes_norm)
            
            node_tensor = sam_wrapper.get_sam_features(masks).to(DEVICE).float().cpu()

        # --- B. EDGE FEATURES (BATCHED CLIP) ---
        edge_indices = []
        text_prompts = []
        batch_crops = []
        
        objects = []
        for k in range(len(boxes)):
            obj = ImageObject(
                label=dataset.get_text_label(labels[k].item()),
                bounding_box=boxes[k].cpu().numpy(),
                mask=masks[k].cpu().numpy(), 
                confidence=1.0
            )
            objects.append(obj)

        pil_image = Image.fromarray(image_rgb)

        for r in relations:
            # relation array is now [s, o, p] from your fixed loader
            if len(r) == 3:
                s_idx, o_idx, p_id = r
            else:
                continue # Skip if malformed

            s_idx, o_idx = s_idx.item(), o_idx.item()
            
            obj_a = objects[s_idx]
            obj_b = objects[o_idx]
            
            # Crop Logic
            x1 = int(min(obj_a.bb[0], obj_b.bb[0]))
            y1 = int(min(obj_a.bb[1], obj_b.bb[1]))
            x2 = int(max(obj_a.bb[2], obj_b.bb[2]))
            y2 = int(max(obj_a.bb[3], obj_b.bb[3]))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            
            if x2 <= x1 or y2 <= y1:
                crop = Image.new('RGB', (224, 224), (0,0,0))
            else:
                crop = pil_image.crop((x1, y1, x2, y2))

            batch_crops.append(crop)
            edge_indices.append([s_idx, o_idx])
            
            # Text Prompt
            subj_txt = objects[s_idx].label
            obj_txt = objects[o_idx].label
            # Handle p_id safely
            pred_txt = dataset.get_text_predicate(p_id.item())
            text_prompts.append(f"a photo of a {subj_txt} {pred_txt} {obj_txt}")

        if len(batch_crops) == 0: continue

        # Run CLIP Batch
        pixel_values = torch.stack([feat_extractor.preprocess(c) for c in batch_crops]).to(DEVICE)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                edge_tensor = feat_extractor.model.encode_image(pixel_values)
                edge_tensor = edge_tensor / edge_tensor.norm(dim=-1, keepdim=True)

        edge_tensor = edge_tensor.float().cpu()
        idx_tensor = torch.tensor(edge_indices)

        # --- C. SAVE ---
        data_packet = {
            "node_feats": node_tensor,
            "edge_feats": edge_tensor,
            "edge_indices": idx_tensor, 
            "text_targets": text_prompts
        }
        
        torch.save(data_packet, save_path)

if __name__ == "__main__":
    preprocess()