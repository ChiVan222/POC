import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
from collections import defaultdict
from transformers import pipeline as hf_pipeline

# GroundingDINO & SAM
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict
from GroundingDINO.groundingdino.util.misc import nested_tensor_from_tensor_list
import GroundingDINO.groundingdino.datasets.transforms as T

# Your SGG Model
from model_decoupled import DecoupledSemanticSGG

# ==========================================================
# 1. CONFIGURATION
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DETECTION_CFG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DETECTION_CKPT = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
FEATURE_CFG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_ovr.py"
FEATURE_CKPT = "weights/vg-pretrain-coco-swinb.pth"
SGG_CKPT = "checkpoints/epoch_50.pth"

VG150_OBJ_CATEGORIES = ['__background__', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

SPATIAL_CLASSES = ["above", "across", "against", "along", "at", "behind", "between", "in", "in front of", "near", "on", "on back of", "over", "under", "with"]
ACTION_CLASSES = ["carrying", "eating", "flying in", "holding", "laying on", "looking at", "lying on", "playing", "riding", "sitting on", "standing on", "using", "walking in", "walking on", "watching", "wearing", "wears"]

BOX_THRESHOLD = 0.35
RELATION_CONF_THRESH = 0.35  
MAX_RELATIONS = 20
DIST_THRESH = 0.3
DEPTH_LAYER_THRESH = 0.1

# ==========================================================
# 2. SYNCED UTILITIES
# ==========================================================
def divide_objects_into_layers(objects, depth_threshold=0.1):
    """Identical layer assignment logic"""
    if len(objects) <= 1: return {}
    sorted_objs = sorted(objects, key=lambda x: x['z'])
    layer_map = {}
    current_idx = 0
    layer_map[sorted_objs[0]['id']] = current_idx
    prev_z = sorted_objs[0]['z']
    for i in range(1, len(sorted_objs)):
        if abs(sorted_objs[i]['z'] - prev_z) > depth_threshold:
            current_idx += 1
        layer_map[sorted_objs[i]['id']] = current_idx
        prev_z = sorted_objs[i]['z']
    return layer_map

def get_geometry_vector(sb, ob, w, h, sd, od):
    """8-dim geometry used in training"""
    sx, sy, sw, sh = sb
    ox, oy, ow, oh = ob
    dx = ((sx + sw/2) - (ox + ow/2)) / w
    dy = ((sy + sh/2) - (oy + oh/2)) / h
    dz = sd - od
    ix1, iy1 = max(sx, ox), max(sy, oy)
    ix2, iy2 = min(sx+sw, ox+ow), min(sy+sh, oy+oh)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = sw*sh + ow*oh - inter
    iou = inter / (union + 1e-6)
    area_ratio = np.log((sw*sh)/(ow*oh + 1e-6))
    dist = np.sqrt(dx**2 + dy**2) / max(w, h)
    return torch.tensor([dx, dy, dz, iou, area_ratio, sd, od, dist], dtype=torch.float32), dist

class FeatureExtractor:
    def __init__(self):
        self.model = load_model(FEATURE_CFG, FEATURE_CKPT).to(DEVICE)
        self.model.eval()
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def get_roi_features(self, image_pil, boxes_xyxy):
        features = []
        for box in boxes_xyxy:
            x1, y1, x2, y2 = map(int, box)
            crop = image_pil.crop((x1, y1, x2, y2))
            img_tensor, _ = self.transform(crop, None)
            samples = nested_tensor_from_tensor_list([img_tensor.to(DEVICE)])
            with torch.no_grad():
                out = self.model.backbone(samples)
                
                # --- FIX FOR ATTRIBUTEERROR: 'LIST' OBJECT HAS NO ATTRIBUTE 'MEAN' ---
                if isinstance(out, dict):
                    raw = out[sorted(out.keys())[-1]]
                elif isinstance(out, (list, tuple)):
                    f_list = out[0]
                    raw = f_list[-1] if isinstance(f_list, (list, tuple)) else f_list
                else:
                    raw = out
                
                t = raw.tensors if hasattr(raw, 'tensors') else raw
                pooled = t.mean(dim=[2, 3])
                features.append(F.normalize(pooled, dim=-1))
        return torch.cat(features) if features else torch.zeros((0, 1024)).to(DEVICE)

    def get_text_prototypes(self, class_names):
        embeds = []
        for name in class_names:
            tok = self.model.tokenizer(name, padding="max_length", return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = self.model.bert(input_ids=tok.input_ids, attention_mask=tok.attention_mask)
                embeds.append(F.normalize(out.last_hidden_state[:, 0], dim=-1))
        return torch.stack(embeds).squeeze(1)

# ==========================================================
# 3. RUN PIPELINE
# ==========================================================
def run_pipeline(image_path):
    image_source = Image.open(image_path).convert("RGB")
    w_img, h_img = image_source.size
    _, image_tensor = load_image(image_path)
    
    # 1. Depth
    depth_pipe = hf_pipeline("depth-estimation", model="LiheYoung/depth-anything-small-hf", device=0)
    depth_out = depth_pipe(image_source)["depth"]
    depth_norm = np.array(depth_out.resize((w_img, h_img))) / 255.0

    # 2. Detect
    detector = load_model(DETECTION_CFG, DETECTION_CKPT).to(DEVICE)
    vg_prompt = " . ".join([c for c in VG150_OBJ_CATEGORIES if c != '__background__'])
    boxes, _, phrases = predict(model=detector, image=image_tensor, caption=vg_prompt, box_threshold=BOX_THRESHOLD, text_threshold=0.25, device=DEVICE)
    
    if len(boxes) == 0: return None
    boxes_xyxy = boxes * torch.Tensor([w_img, h_img, w_img, h_img])
    boxes_xyxy[:, :2] -= boxes_xyxy[:, 2:] / 2
    boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]

    # 3. Process Objects
    objects = []
    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[i].tolist()
        bx, by, bw, bh = int(max(0, x1)), int(max(0, y1)), int(max(1, x2-x1)), int(max(1, y2-y1))
        depth_patch = depth_norm[by:by+bh, bx:bx+bw]
        z = float(np.median(depth_patch)) if depth_patch.size > 0 else 0.5
        objects.append({'id': i, 'box': [bx, by, bw, bh], 'z': z, 'label': phrases[i]})

    layer_map = divide_objects_into_layers(objects, depth_threshold=DEPTH_LAYER_THRESH)

    # 4. SGG Inference
    feat_extractor = FeatureExtractor()
    vis_feats = feat_extractor.get_roi_features(image_source, boxes_xyxy)
    s_anchors = feat_extractor.get_text_prototypes(["__background__"] + SPATIAL_CLASSES)
    a_anchors = feat_extractor.get_text_prototypes(["__background__"] + ACTION_CLASSES)

    sgg_model = DecoupledSemanticSGG(vis_dim=1024, geo_dim=8, embed_dim=768).to(DEVICE)
    sgg_model.load_state_dict(torch.load(SGG_CKPT, map_location=DEVICE))
    sgg_model.eval()
    
    all_predictions = []
    for i in range(len(objects)):
        for j in range(len(objects)):
            if i == j: continue
            
            subj, obj = objects[i], objects[j]
            geo_vec, dist = get_geometry_vector(subj['box'], obj['box'], w_img, h_img, subj['z'], obj['z'])
            
            # Filtering Synced with preprocess.py
            bx1, by1, bw1, bh1 = subj['box']; bx2, by2, bw2, bh2 = obj['box']
            is_overlap = (max(bx1, bx2) < min(bx1+bw1, bx2+bw2) and max(by1, by2) < min(by1+bh1, by2+bh2))
            same_layer = (layer_map.get(subj['id']) == layer_map.get(obj['id']))
            
            if not (is_overlap or (same_layer and dist < DIST_THRESH)):
                continue
            
            with torch.no_grad():
                sl, al, *_ = sgg_model(geo_vec.to(DEVICE).unsqueeze(0), vis_feats[i].unsqueeze(0), s_anchors, a_anchors)
            
            # Calibrate: Apply Temp (0.07) and Skip Background
            s_prob = F.softmax(sl, dim=-1)[:, 1:] 
            a_prob = F.softmax(al / 0.07, dim=-1)[:, 1:] 
            
            s_val, s_id = s_prob.max(1); a_val, a_id = a_prob.max(1)
            
            if s_val > RELATION_CONF_THRESH:
                all_predictions.append({'s': subj['label'], 'p': SPATIAL_CLASSES[s_id.item()], 'o': obj['label'], 'score': s_val.item(), 'boxes': (subj['box'], obj['box'])})
            if a_val > RELATION_CONF_THRESH:
                all_predictions.append({'s': subj['label'], 'p': ACTION_CLASSES[a_id.item()], 'o': obj['label'], 'score': a_val.item(), 'boxes': (subj['box'], obj['box'])})

    return image_source, sorted(all_predictions, key=lambda x: x['score'], reverse=True)[:MAX_RELATIONS]

# ==========================================================
# 4. VISUALIZATION
# ==========================================================
def visualize_pipeline(img_pil, relations):
    draw = ImageDraw.Draw(img_pil)
    print(f"\n{'SUBJECT':<15} | {'PREDICATE':<15} | {'OBJECT':<15} | {'CONF'}")
    print("-" * 65)
    for rel in relations:
        print(f"{rel['s']:<15} | {rel['p']:<15} | {rel['o']:<15} | {rel['score']:.4f}")
        sb, ob = rel['boxes']
        s_c = (sb[0]+sb[2]/2, sb[1]+sb[3]/2); o_c = (ob[0]+ob[2]/2, ob[1]+ob[3]/2)
        draw.line([s_c, o_c], fill="cyan", width=3)
        draw.text(o_c, rel['p'], fill="yellow")
    img_pil.show()

if __name__ == "__main__":
    res = run_pipeline("vg_data/VG_100K/300.jpg")
    if res: visualize_pipeline(res[0], res[1])