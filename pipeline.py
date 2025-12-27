from groundingdino.util.inference import load_model, load_image,predict, annotate   
from segment_anything import sam_model_registry, SamPredictor
import torch
import cv2
import numpy as np
import os
from PIL import Image
import clip
import matplotlib.pyplot as plt
from typing import List
import networkx as nx
from model.GNN import GNN
from blip2_interaction import BLIP2Interaction

DEVICE = "cuda"
IMAGE_PATH = "testImgs/skateboarders.jpeg"
DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_CHECKPOINT = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT = "GroundingDINO/weights/sam_vit_b_01ec64.pth" 
SAM_TYPE = "vit_b" 
DEVICE = "cuda"
OUTPUT_DIR = "outputs"
VG150_OBJ_CATEGORIES = [
    'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 
    'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 
    'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup', 
    'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 
    'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 
    'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 
    'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men', 
    'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 
    'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 
    'post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 
    'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 
    'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 
    'towel', 'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 
    'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra'
]
VG150_PREDICATES = [
    "above", "across", "against", "along", "and", "at", "attached to", "behind", 
    "belonging to", "between", "carrying", "covered in", "covering", "eating", 
    "flying in", "for", "from", "growing on", "hanging from", "has", "holding", 
    "in", "in front of", "laying on", "looking at", "lying on", "made of", 
    "mounted on", "near", "of", "on", "on back of", "over", "painted on", 
    "parked on", "part of", "playing", "riding", "says", "sitting on", 
    "standing on", "to", "under", "using", "walking in", "walking on", 
    "watching", "wearing", "wears", "with", "no relation"
]
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label, color='white', fontsize=10, backgroundcolor='green')





class ImageObject:
    def __init__(self, label, bounding_box, mask, confidence, features=None):
        self.label = label          
        self.bb = bounding_box     
        self.mask = mask            
        self.confidence = confidence 
        self.features = features    
        self.sam_feats = None 
    def __repr__(self):
        return f"ImageObject(Label: {self.label}, Conf: {self.confidence:.2f})"

class GroundingDinoWrapper:
    def __init__(self, DINO_CONFIG, DINO_CKPT, BOX_TRESHOLD=0.35, TEXT_TRESHOLD=0.25):
        print("Initiating Dino \n")
        self.model = load_model(DINO_CONFIG, DINO_CKPT)
        self.bth = BOX_TRESHOLD
        self.tth = TEXT_TRESHOLD

    def process(self, img_path, text_prompt):
        image_source, image = load_image(img_path)
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=text_prompt,
            box_threshold=self.bth,
            text_threshold=self.tth
        )
        return image_source, boxes, logits, phrases

class SAMWrapper:
    def __init__(self, SAM_TYPE, SAM_CKPT):
        print("Initiating SAM \n")
        self.model = sam_model_registry[SAM_TYPE](checkpoint=SAM_CKPT).to(DEVICE)
        self.predictor = SamPredictor(self.model)

    # In pipeline.py

class SAMWrapper:
    def __init__(self, SAM_TYPE, SAM_CKPT):
        print("Initiating SAM \n")
        self.model = sam_model_registry[SAM_TYPE](checkpoint=SAM_CKPT).to(DEVICE)
        self.predictor = SamPredictor(self.model)

    def process(self, img_path, boxes):
        image_cv2 = cv2.imread(img_path)
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        H, W, _ = image_cv2.shape

        # Fix device mismatch crash
        scale = torch.tensor([W, H, W, H], device=boxes.device)
        boxes_xyxy = boxes * scale
        boxes_xyxy[:, :2] -= boxes_xyxy[:, 2:] / 2
        boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]

        self.predictor.set_image(image_cv2)
        
        # Explicitly move to device
        boxes_cuda = boxes_xyxy.to(DEVICE)
        
        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_cuda, image_cv2.shape[:2])
        
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        masks = masks.squeeze(1)
        return masks, boxes_xyxy

    def get_sam_features(self, masks): 
        """
        Vectorized extraction of SAM features.
        Speeds up processing by removing the for-loop.
        """
        features = self.predictor.get_image_embedding() # [1, 256, 64, 64]
        
        # Ensure masks are on GPU and float
        if not isinstance(masks, torch.Tensor):
             masks = torch.from_numpy(masks)
        masks = masks.to(DEVICE).float()
        
        # 1. Resize ALL masks at once to [N, 1, 64, 64]
        # We assume masks is [N, H, W], unsqueeze to [N, 1, H, W]
        masks_resized = torch.nn.functional.interpolate(
            masks.unsqueeze(1), 
            size=(features.shape[-2], features.shape[-1]), 
            mode='nearest'
        ) # [N, 1, 64, 64]
        
        # 2. Broadcast multiply: [1, 256, 64, 64] * [N, 1, 64, 64] = [N, 256, 64, 64]
        masked_feats = features * masks_resized
        
        # 3. Global Average Pooling (Sum over H,W dims)
        # Sum over last two dims (-1, -2)
        sum_feats = masked_feats.sum(dim=(-1, -2))
        area = masks_resized.sum(dim=(-1, -2)) + 1e-6
        
        avg_feats = sum_feats / area # [N, 256]
        
        return avg_feats
    
    def get_sam_features(self, masks): 
        features = self.predictor.get_image_embedding()
        object_sam_vectors = []
        
        # FIX 3: Batch processing for speed (Optional but recommended)
        # Ensure masks is a Tensor on the correct device
        if not isinstance(masks, torch.Tensor):
             masks = torch.from_numpy(masks)
        masks = masks.to(DEVICE).float()
             
        for mask in masks:
            # Resize mask to match feature map (64x64)
            mask_input = mask.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
            
            mask_small = torch.nn.functional.interpolate(
                mask_input, 
                size=(features.shape[-2], features.shape[-1]), 
                mode='nearest'
            ).squeeze()          
            
            masked_feat = features.squeeze(0) * mask_small
            avg_feat = masked_feat.sum(dim=(1, 2)) / (mask_small.sum() + 1e-6)
            object_sam_vectors.append(avg_feat) 
            
        if len(object_sam_vectors) == 0:
            return torch.zeros((0, 256), device=DEVICE)
            
        return torch.stack(object_sam_vectors)
class FeatureExtractor: 
    def __init__(self, device="cuda"): 
        self.device = device  
        self.model, self.preprocess = clip.load("ViT-L/14",device=self.device)  
    def extract_object_features(self,image_source, object_list): 
        if not isinstance(image_source, Image.Image):
            image_source = Image.fromarray(image_source)       
        print(f"Extracting features for {len(object_list)} objects...")
        w,h = image_source.size
        valid_objects = [ ]
        batch_crops = []
        for obj in object_list:  
            x1,y1,x2,y2 = map(int, obj.bb)
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w,x2), min(h,y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                obj.features = torch.zeros(768, device=self.device)
                continue
            object_crop = image_source.crop((x1,y1,x2,y2))
            if obj.mask is not None:  
                obj_mask = obj.mask[y1:y2,x1:x2]
                if obj_mask.shape[0] > 0 and obj_mask.shape[1]>  0 :
                    crop_arr = np.array(object_crop)
                    if obj_mask.shape[:2] != crop_arr.shape[:2]:
                        obj_mask = cv2.resize(obj_mask.astype(float), 
                                            (crop_arr.shape[1], crop_arr.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
                    if len(obj_mask.shape) == 2:
                        obj_mask = obj_mask[:, :, None]
                    masked_arr = crop_arr * obj_mask
                    object_crop = Image.fromarray(masked_arr.astype(np.uint8))
            batch_crops.append(object_crop)
            valid_objects.append(obj)
        if not batch_crops: 
            return
        processed_tensors = [self.preprocess(crop) for crop in batch_crops]
        batch_input = torch.stack(processed_tensors).to(self.device)
        print(f"Running CLIP on batch shape: {batch_input.shape}")
        with torch.no_grad():
            batch_features = self.model.encode_image(batch_input)
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
        for obj, feat in zip(valid_objects, batch_features):
            obj.features = feat 
    def get_union_feature(self, image_source, obj_a, obj_b):
        """
        Extracts the union region and applies the union mask to black out background.
        Requires obj_a and obj_b to be ImageObject instances (with .mask).
        """
        x1 = int(min(obj_a.bb[0], obj_b.bb[0]))
        y1 = int(min(obj_a.bb[1], obj_b.bb[1]))
        x2 = int(max(obj_a.bb[2], obj_b.bb[2]))
        y2 = int(max(obj_a.bb[3], obj_b.bb[3]))
        if not isinstance(image_source, Image.Image):
            image_source = Image.fromarray(image_source)
        W, H = image_source.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return torch.zeros(768, device=self.device) 
            
        union_crop = image_source.crop((x1, y1, x2, y2))
      
        if obj_a.mask is not None and obj_b.mask is not None:
            mask_a = obj_a.mask[y1:y2, x1:x2]
            mask_b = obj_b.mask[y1:y2, x1:x2]
            
            union_mask = np.logical_or(mask_a, mask_b).astype(np.uint8)
            
            crop_w, crop_h = union_crop.size
            if union_mask.shape != (crop_h, crop_w):
                union_mask = cv2.resize(union_mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
            
            union_mask_rgb = np.stack([union_mask]*3, axis=-1)
            
            crop_arr = np.array(union_crop)
            masked_arr = crop_arr * union_mask_rgb
            
            union_crop = Image.fromarray(masked_arr)

        inputs = self.preprocess(union_crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(inputs)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            
        return feat.squeeze(0)
class RelationProcessor:
    def __init__(self, clip_model, clip_preprocess, device="cuda"):
        self.clip_model = clip_model
        self.preprocess = clip_preprocess
        self.device = device
        self.relations = VG150_PREDICATES
    def get_union_crop(self, image_np, box1, box2):
        """
        Crops the image to contain both objects + padding.
        """
        x1_a, y1_a, x2_a, y2_a = box1
        x1_b, y1_b, x2_b, y2_b = box2

        # Find Union coordinates
        x1 = min(x1_a, x1_b)
        y1 = min(y1_a, y1_b)
        x2 = max(x2_a, x2_b)
        y2 = max(y2_a, y2_b)
        
        # Add padding
        H, W, _ = image_np.shape
        pad = 20
        x1 = max(0, int(x1) - pad)
        y1 = max(0, int(y1) - pad)
        x2 = min(W, int(x2) + pad)
        y2 = min(H, int(y2) + pad)

        return image_np[y1:y2, x1:x2]

    def predict_relations(self, image_source, objects):
        results = []
        print(f"Analyzing detailed relations...")

        # Ensure numpy for slicing
        if isinstance(image_source, Image.Image):
            image_np = np.array(image_source)
        else:
            image_np = image_source

        for i, subj in enumerate(objects):
            for j, obj in enumerate(objects):
                if i == j: continue 

                # 1. OPTIONAL: Geometric Filter (Skip if too far)
                # dist = ... (implement if needed)

                # 2. PREPARE TEXT PROMPTS (OpenAI CLIP format)
                prompts = [f"a photo of a {subj.label} {rel} {obj.label}" for rel in self.relations]
                text_tokens = clip.tokenize(prompts).to(self.device)

                # 3. PREPARE IMAGE CROP
                union_crop = self.get_union_crop(image_np, subj.bb, obj.bb)
                if union_crop.size == 0 or union_crop.shape[0] == 0 or union_crop.shape[1] == 0:
                    continue
                
                # Preprocess image (OpenAI CLIP format)
                # self.preprocess expects PIL, returns Tensor (3, 224, 224)
                image_input = self.preprocess(Image.fromarray(union_crop)).unsqueeze(0).to(self.device)

                # 4. INFERENCE
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_input)
                    text_features = self.clip_model.encode_text(text_tokens)

                    # Normalize
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    # Similarity (Dot product)
                    # OpenAI CLIP uses a scaling factor (logit_scale), usually 100.0 manually applied
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    
                    # Get Top-1
                    values, indices = similarity[0].topk(1)
                    best_idx = indices.item()
                    best_score = values.item()
                    best_rel = self.relations[best_idx]

                # 5. THRESHOLD
                if best_score > 0.25: # Slightly higher threshold for better precision
                    print(f"Match: {subj.label} [{best_rel}] {obj.label} (Score: {best_score:.2f})")
                    results.append((subj.label, best_rel, obj.label))
                    
        return results
    def _precompute_text_features(self):
        prompts = []
        for r in self.relations:
            if r.lower() in ["no", "no relation", "none"]:
                prompts.append("a photo of two objects with no relation")
            else:
                prompts.append(f"a photo of something {r} something")
                
        text_tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            feats = self.clip_model.encode_text(text_tokens)
            self.text_features = feats / feats.norm(dim=-1, keepdim=True)

    def get_text_features(self):
        if not hasattr(self, 'text_features') or self.text_features is None:
            self._precompute_text_features()
        return self.text_features


class SceneGraph:  
    def __init__(self, image_source=None, objects=None, relations=None): 
        self.img = image_source
        self.objects = objects if objects is not None else []
        self.relations = relations if relations is not None else []
    def set_img(self, img_source): 
         self.img  = img_source  
    def visualize(self, ax=None, SEGMENT_FLAG=True):
        if self.img is None: return
        
        if ax is None:
            plt.figure(figsize=(10, 10))
            ax = plt.gca()
        
        ax.imshow(self.img)
        for obj in self.objects:
            if SEGMENT_FLAG and obj.mask is not None:
                show_mask(obj.mask, ax, random_color=True)
            show_box(obj.bb, ax, f"{obj.label} {obj.confidence:.2f}")
        ax.axis('off')
    def visualize_scene_graph(self, SEGMENT_FLAG=True):
        if self.img is None:
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        # 1. Draw base image + objects
        self.visualize(ax=ax, SEGMENT_FLAG=SEGMENT_FLAG)

        # 2. Build graph + positions (pixel space)
        G = nx.DiGraph()
        pos = {}
        node_labels = {}

        for idx, obj in enumerate(self.objects):
            x1, y1, x2, y2 = obj.bb
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            G.add_node(idx)
            pos[idx] = (cx, cy)
            node_labels[idx] = obj.label

        for subj_str, pred, obj_str in self.relations:
            subj_idx = None
            obj_idx = None

            for idx, obj in enumerate(self.objects):
                if obj.label == subj_str and subj_idx is None:
                    subj_idx = idx
                if obj.label == obj_str and obj_idx is None:
                    obj_idx = idx

            if subj_idx is not None and obj_idx is not None:
                G.add_edge(subj_idx, obj_idx, label=pred)

        nx.draw_networkx_nodes(
            G, pos,
            node_size=500,
            node_color="lime",
            edgecolors="black",
            alpha=0.8,
            ax=ax
        )

        nx.draw_networkx_labels(
            G, pos,
            labels=node_labels,
            font_size=9,
            font_weight="bold",
            font_color="black",
            ax=ax
        )

        # 5. Draw relations (arrows)
        nx.draw_networkx_edges(
            G, pos,
            ax=ax,
            arrows=True,
            arrowstyle="->",
            arrowsize=20,
            edge_color="cyan",
            width=2,
            alpha=0.7,
            connectionstyle="arc3,rad=0.1" 
        )

        # 6. Draw relation labels
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=8,
            font_color="white",
            ax=ax,
            bbox=dict(facecolor="red", alpha=0.6, edgecolor="none", pad=1)
        )

        ax.set_title("Scene Graph Overlay", fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.show()
        plt.savefig("output")



class PipeLine:

    def __init__(self):
        self.dino_wrapper = GroundingDinoWrapper(DINO_CONFIG=DINO_CONFIG, DINO_CKPT=DINO_CHECKPOINT)
        self.sam = SAMWrapper(SAM_TYPE=SAM_TYPE, SAM_CKPT=SAM_CHECKPOINT)
        self.feat =  FeatureExtractor()
        self.sg = SceneGraph()
        checkpoint_path = "checkpoints/gnn_epoch_79.pth" 
        self.gnn = GNN()
        self.blip2 = BLIP2Interaction(device=DEVICE)
        self._blip_cache = {}

        if os.path.exists(checkpoint_path):
            print(f"Loading GNN checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            
            # Handle the dictionary format saved by train_fast.py
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.gnn.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Fallback for old-style checkpoints (weights only)
                self.gnn.load_state_dict(checkpoint)
                
        else:
            print(f"⚠️ WARNING: No checkpoint found at {checkpoint_path}. Using random GNN weights!")
        self.gnn.eval() 
        self.gnn = self.gnn.to(DEVICE)
    def process(self, img_path, text_prompt = None, SEGMENT_FLAG = True):
        if text_prompt is None:
            text_prompt = " . ".join(VG150_OBJ_CATEGORIES)
        source, boxes_norm, logits, phrases = self.dino_wrapper.process(img_path, text_prompt)
        self.sg.img  = source
        if len(boxes_norm) == 0:
            print(f"No objects detected in {img_path}. Skipping SAM.")
            return []
        if SEGMENT_FLAG: 
            masks, boxes_abs = self.sam.process(img_path, boxes_norm)
            all_sam_feats = self.sam.get_sam_features(masks)
        else:
            image_cv2 = cv2.imread(img_path)
            image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
            H, W, _ = image_cv2.shape
            boxes_abs = boxes_norm * torch.Tensor([W, H, W, H])
            boxes_abs[:, :2] -= boxes_abs[:, 2:] / 2  
            boxes_abs[:, 2:] += boxes_abs[:, :2]    
            masks = [None] * len(boxes_abs)
            all_sam_feats = torch.zeros((len(boxes_abs), 256)).to(DEVICE)
        self.sg.objects = [] 
        
        for i,(mask, box, label, conf) in enumerate(zip(masks, boxes_abs, phrases, logits)):
            final_mask = mask.cpu().numpy() if mask is not None else None
            obj = ImageObject(
                label=label,
                bounding_box=box.cpu().numpy(),
                mask=final_mask, 
                confidence=conf.item()
            )
            obj.sam_feats = all_sam_feats[i]
            self.sg.objects.append(obj)
        values = None
        if len(self.sg.objects) > 0:
            print("Step 2: Extracting Node Features...")
            edge_feats_list = []
            edge_indices_list = []

            for i, subj in enumerate(self.sg.objects):
                for j, obj in enumerate(self.sg.objects):
                    if i == j: continue
                    
                    obj_a = self.sg.objects[i]
                    obj_b = self.sg.objects[j]

                    # Calculate coordinates
                    x1 = int(min(obj_a.bb[0], obj_b.bb[0]))
                    y1 = int(min(obj_a.bb[1], obj_b.bb[1]))
                    x2 = int(max(obj_a.bb[2], obj_b.bb[2]))
                    y2 = int(max(obj_a.bb[3], obj_b.bb[3]))

                    # Clamp to image size
                    if not isinstance(source, Image.Image):
                         image_source = Image.fromarray(source)
                    W,H  = image_source.size
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)
                    
                    
                    # Create the variable that was missing!
                    union_crop_img = image_source.crop((x1, y1, x2, y2))

                    # 2. RUN SPECIFIC FILTER
                    subj_lbl = self.sg.objects[i].label
                    obj_lbl = self.sg.objects[j].label
                    if subj_lbl in ["hair", "cap", "hat", "jean", "shirt", "pant"]:
                        continue  # body parts should not be subjects

                                            # Safety Check
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    is_interacting = self.blip2.is_interacting(
                            union_crop_img,
                            subj_lbl,
                            obj_lbl
                        )
                    if not is_interacting:
                        continue
                    union_feat = self.feat.get_union_feature(source,obj_a= obj_a,obj_b=obj_b)
                    edge_feats_list.append(union_feat)
                    edge_indices_list.append([i, j])


            if len(edge_feats_list) > 0:
                node_tensor = all_sam_feats.to(DEVICE).float()                    # [N, 256]
                edge_tensor = torch.stack(edge_feats_list).to(DEVICE).float()      # [E, 768]
                idx_tensor = torch.tensor(edge_indices_list).to(DEVICE)    # [E, 2]
                with torch.no_grad():
                    visual_concepts = self.gnn(node_tensor, edge_tensor, idx_tensor)

                if not hasattr(self, 'rel_processor'):
                     self.rel_processor = RelationProcessor(self.feat.model, self.feat.preprocess)
                text_feats = self.rel_processor.get_text_features()
                visual_norm = visual_concepts / visual_concepts.norm(dim=-1, keepdim=True)
                text_norm = text_feats / text_feats.norm(dim=-1, keepdim=True)
                similarity = (100.0 * visual_norm @ text_norm.float().T).softmax(dim=-1)
                values, indices = similarity.topk(1)
            if len(edge_feats_list) == 0:
                print("⚠️ BLIP rejected all edges — disabling BLIP for this image")
                for i, subj in enumerate(self.sg.objects):
                    for j, obj in enumerate(self.sg.objects):
                        if i != j:
                            union_feat = self.feat.get_union_feature(
                                source, subj, obj
                            )
                            edge_feats_list.append(union_feat)
                            edge_indices_list.append([i, j])

            print("Step 3: Decoding GNN Predictions...")
            for k, (val, idx) in enumerate(zip(values, indices)):
                rel_idx = idx.item()
                score = val.item()
                
                subj_i, obj_j = edge_indices_list[k]
                subj_lbl = self.sg.objects[subj_i].label
                obj_lbl = self.sg.objects[obj_j].label
                rel_name = self.rel_processor.relations[rel_idx]
                
                if rel_name not in ["No", "no relation", "background"] and score > 0.7:
                     print(f"Match: {subj_lbl} [{rel_name}] {obj_lbl} ({score:.2f})")
                     self.sg.relations.append((subj_lbl, rel_name, obj_lbl))
        return self.sg
        

if __name__=="__main__":
    p = PipeLine()
    sg = p.process("testImgs/skateboarders.jpeg",SEGMENT_FLAG=1)    
    sg.visualize_scene_graph()



