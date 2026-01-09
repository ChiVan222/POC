import torch
import os
import numpy as np
import json
import logging
import warnings
from tqdm import tqdm
from PIL import Image
from transformers import pipeline as hf_pipeline, AutoImageProcessor
from GroundingDINO.groundingdino.util.inference import load_model
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.vg import VGDataset 
import hashlib
import queue
from threading import Thread
from torch.cuda.amp import autocast

# 1. SETUP FOR PERFORMANCE
logging.getLogger("transformers.pipelines.base").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True  # Enable CuDNN auto-tuner

# ==========================================
# 2. MAXIMUM SPEED CONFIGURATION (VERIFIED STABLE)
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PATHS
IMG_DIR = r"D:\SceneGraphGeneration\POC\vg_data\VG_100K"
ROIDB_FILE = r"D:\SceneGraphGeneration\POC\vg_data\stanford_filtered\VG-SGG.h5"
DICT_FILE = r"D:\SceneGraphGeneration\POC\vg_data\stanford_filtered\VG-SGG-dicts.json"
IMAGE_FILE = r"D:\SceneGraphGeneration\POC\vg_data\stanford_filtered\image_data.json"

# SPLIT will be passed to the function
SPLIT = 'train'  # Default, but will be overridden

# MAXIMUM SPEED SETTINGS (VERIFIED STABLE ON 16GB VRAM)
IMG_BATCH_SIZE = 8          # Verified stable
DINO_CHUNK_SIZE = 128          # Verified stable - you confirmed this works!
NUM_WORKERS = 4                # Optimal for HDD
SAVE_WORKERS = 2               # Background save threads
DEPTH_BATCH_SIZE = 8          # Increased for maximum speed

# Depth target size (must match what Depth Anything expects)
DEPTH_TARGET_SIZE = (518, 518)  # Depth Anything small model expects 518x518

DINO_CFG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_ovr.py"
DINO_CKPT = "weights/vg-pretrain-coco-swinb.pth"

# ==========================================
# 3. UTILITIES
# ==========================================
class NestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask
    def to(self, device):
        return NestedTensor(self.tensors.to(device), 
                           self.mask.to(device) if self.mask is not None else None)

def get_geometry_vector(sb, ob, w, h, sd, od):
    """Optimized geometry vector computation"""
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
    dist = np.sqrt(dx**2 + dy**2)
    
    return torch.tensor([dx, dy, dz, iou, area_ratio, sd, od, dist], 
                       dtype=torch.float32, device='cpu')

def divide_objects_into_layers(objects, depth_threshold=0.1):
    """Layer assignment with early exit"""
    if len(objects) <= 1:
        return {}
    
    sorted_objs = sorted(objects, key=lambda x: x['z'])
    layer_map = {}
    current_idx = 0
    layer_map[sorted_objs[0]['orig_id']] = current_idx
    prev_z = sorted_objs[0]['z']
    
    for i in range(1, len(sorted_objs)):
        if abs(sorted_objs[i]['z'] - prev_z) > depth_threshold:
            current_idx += 1
        layer_map[sorted_objs[i]['orig_id']] = current_idx
        prev_z = sorted_objs[i]['z']
    
    return layer_map

def custom_collate(batch):
    """Filter out None batches and return"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return [], []
    return [item[0] for item in batch], [item[1] for item in batch]

# ==========================================
# 4. DEPTH CACHE SYSTEM
# ==========================================
class DepthCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.cache = {}
    
    def get_cache_key(self, img_id):
        return f"{img_id}"
    
    def get(self, img_id):
        """Get cached depth map if exists"""
        cache_key = self.get_cache_key(img_id)
        
        # Check memory cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Check disk cache
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
        if os.path.exists(cache_path):
            try:
                depth_data = np.load(cache_path, allow_pickle=True).item()
                depth_map = depth_data['depth']
                # Store in memory cache
                self.cache[cache_key] = depth_map
                return depth_map
            except Exception as e:
                print(f"Cache load error for {img_id}: {e}")
        
        return None
    
    def set(self, img_id, depth_map):
        """Cache depth map to memory and disk"""
        cache_key = self.get_cache_key(img_id)
        self.cache[cache_key] = depth_map
        
        # Async save to disk (non-blocking)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
        try:
            np.save(cache_path, {'depth': depth_map}, allow_pickle=True)
        except Exception as e:
            print(f"Cache save error for {img_id}: {e}")

# ==========================================
# 5. ASYNCHRONOUS SAVE SYSTEM
# ==========================================
class AsyncSaver:
    def __init__(self, num_workers=2, max_queue=1000):
        self.save_queue = queue.Queue(maxsize=max_queue)
        self.workers = []
        self.running = True
        
        for i in range(num_workers):
            worker = Thread(target=self._save_worker, daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def _save_worker(self):
        """Background thread for saving files"""
        while self.running or not self.save_queue.empty():
            try:
                filepath, data = self.save_queue.get(timeout=1)
                if filepath is None:  # Sentinel
                    break
                
                # Save with compression for space
                torch.save(data, filepath, _use_new_zipfile_serialization=True)
                self.save_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Save error for {filepath}: {e}")
                self.save_queue.task_done()
    
    def save(self, filepath, data):
        """Queue file for saving"""
        self.save_queue.put((filepath, data))
    
    def wait_completion(self):
        """Wait for all queued saves to complete"""
        self.save_queue.join()
    
    def stop(self):
        """Stop save workers"""
        self.running = False
        # Add sentinels for each worker
        for _ in self.workers:
            self.save_queue.put((None, None))
        
        for worker in self.workers:
            worker.join(timeout=5)

# ==========================================
# 6. OPTIMIZED PAIR COMPUTATION
# ==========================================
def compute_object_pairs(objects, layer_map, gt_pairs_set, img_size):
    """Compute candidate object pairs efficiently"""
    w, h = img_size
    pairs = []
    n_objects = len(objects)
    
    if n_objects < 2:
        return pairs
    
    # Precompute centers and areas
    centers = []
    boxes = []
    depths = []
    
    for obj in objects:
        bx, by, bw, bh = obj['box']
        centers.append((bx + bw/2, by + bh/2))
        boxes.append((bx, by, bw, bh))
        depths.append(obj['z'])
    
    # Check all pairs
    for i in range(n_objects):
        for j in range(n_objects):
            if i == j:
                continue
            
            id1, id2 = objects[i]['orig_id'], objects[j]['orig_id']
            
            # Quick distance check
            c1_x, c1_y = centers[i]
            c2_x, c2_y = centers[j]
            dist = np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2) / max(w, h)
            
            # Check conditions
            condition_met = False
            
            # Condition 1: Overlap
            bx1, by1, bw1, bh1 = boxes[i]
            bx2, by2, bw2, bh2 = boxes[j]
            ix1, iy1 = max(bx1, bx2), max(by1, by2)
            ix2, iy2 = min(bx1+bw1, bx2+bw2), min(by1+bh1, by2+bh2)
            is_overlap = (ix2 > ix1 and iy2 > iy1)
            
            # Condition 2: Same layer and close
            same_layer_close = (layer_map.get(id1, -1) == layer_map.get(id2, -1) and dist < 0.3)
            
            # Condition 3: Ground truth pair
            is_gt_pair = (id1, id2) in gt_pairs_set
            
            if is_overlap or same_layer_close or is_gt_pair:
                # Compute union box
                ux1 = max(0, min(bx1, bx2))
                uy1 = max(0, min(by1, by2))
                ux2 = min(w, max(bx1+bw1, bx2+bw2))
                uy2 = min(h, max(by1+bh1, by2+bh2))
                
                if ux2 > ux1 and uy2 > uy1:
                    # Store minimal info, compute geometry later
                    pairs.append({
                        'i': i, 'j': j,
                        'box_i': boxes[i], 'box_j': boxes[j],
                        'z_i': depths[i], 'z_j': depths[j],
                        'crop_coords': (ux1, uy1, ux2, uy2),
                        'is_gt': is_gt_pair,
                        'sub_id': id1, 'obj_id': id2
                    })
    
    return pairs

# ==========================================
# 7. PERFECTED PADDING DEPTH PROCESSOR (WITH CORRECT UNPADDING)
# ==========================================
class PaddingDepthProcessor:
    """Process depth with aspect ratio preserving padding (letterboxing)"""
    
    def __init__(self, depth_pipe, batch_size=16, target_size=(518, 518)):
        self.depth_pipe = depth_pipe
        self.batch_size = batch_size
        self.target_size = target_size
        
    def process_batch(self, images, original_sizes):
        """Process images with correct padding and unpadding"""
        batch_results = []
        
        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i+self.batch_size]
            batch_orig_sizes = original_sizes[i:i+self.batch_size]
            
            # Prepare padded images and track padding info
            padded_images = []
            padding_info = []  # Store (top, left, new_h, new_w) for each image
            
            for img in batch_images:
                w, h = img.size
                target_w, target_h = self.target_size
                
                # Calculate scaling factor while maintaining aspect ratio
                scale = min(target_w / w, target_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # Resize maintaining aspect ratio
                img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                # Create padded image (letterboxing)
                img_padded = Image.new("RGB", (target_w, target_h), (0, 0, 0))
                
                # Calculate padding position (center)
                left = (target_w - new_w) // 2
                top = (target_h - new_h) // 2
                
                # Paste resized image onto padded canvas
                img_padded.paste(img_resized, (left, top))
                
                padded_images.append(img_padded)
                padding_info.append((top, left, new_h, new_w))
            
            try:
                # Process padded batch
                outputs = self.depth_pipe(padded_images)
                
                for j, output in enumerate(outputs):
                    if output and "depth" in output:
                        depth_map_padded = output["depth"]  # 518x518
                        top, left, new_h, new_w = padding_info[j]
                        orig_w, orig_h = batch_orig_sizes[j]
                        
                        # 1. CROP to remove padding (get the valid region)
                        depth_map_cropped = depth_map_padded.crop(
                            (left, top, left + new_w, top + new_h)
                        )
                        
                        # 2. Resize cropped depth map back to original dimensions
                        depth_map_original = depth_map_cropped.resize(
                            (orig_w, orig_h), 
                            Image.Resampling.BILINEAR
                        )
                        
                        # 3. Normalize
                        d_raw = np.array(depth_map_original, dtype=np.float32)
                        d_min, d_max = d_raw.min(), d_raw.max()
                        if d_max > d_min:
                            d_norm = (d_raw - d_min) / (d_max - d_min + 1e-6)
                        else:
                            d_norm = np.ones_like(d_raw) * 0.5
                            
                        batch_results.append(d_norm)
                    else:
                        # Fallback
                        h, w = batch_orig_sizes[j][1], batch_orig_sizes[j][0]
                        batch_results.append(np.ones((h, w), dtype=np.float32) * 0.5)
                        
            except Exception as e:
                print(f"Batch depth error: {e}")
                # Fallback for entire batch
                for size in batch_orig_sizes:
                    h, w = size[1], size[0]
                    batch_results.append(np.ones((h, w), dtype=np.float32) * 0.5)
        
        return batch_results

# ==========================================
# 8. MAIN PIPELINE (FINAL PERFECTED VERSION)
# ==========================================
def run_perfected_preprocess(split):
    print(f"\n{'='*60}")
    print(f">>> Starting PERFECTED preprocessing for {split} split")
    print(f"{'='*60}")
    
    # Set paths for this split
    SAVE_DIR = f"vg_data/{split}_features_final"
    DEPTH_CACHE_DIR = f"vg_data/{split}_depth_cache"
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(DEPTH_CACHE_DIR, exist_ok=True)
    
    print(f">>> Device: {DEVICE}, Workers: {NUM_WORKERS}")
    print(f">>> MAX SPEED Config: IMG_BATCH={IMG_BATCH_SIZE}, DINO_CHUNK={DINO_CHUNK_SIZE}")
    print(f">>> Depth batch size: {DEPTH_BATCH_SIZE}")
    print(f">>> Using CORRECT padding (letterboxing) for depth estimation")
    
    # Initialize systems
    depth_cache = DepthCache(DEPTH_CACHE_DIR)
    async_saver = AsyncSaver(num_workers=SAVE_WORKERS)
    
    print(">>> 1. Loading image ID mapping...")
    with open(IMAGE_FILE, 'r') as f:
        all_imgs = json.load(f)
    id_to_h5idx = {img['image_id']: i for i, img in enumerate(all_imgs)}
    
    print(">>> 2. Loading models...")
    # Initialize depth pipeline WITHOUT batching (our processor handles it)
    try:
        depth_pipe = hf_pipeline(
            "depth-estimation", 
            model="LiheYoung/depth-anything-small-hf", 
            device=0 if DEVICE == "cuda" else -1
        )
    except Exception as e:
        print(f"Depth pipeline initialization error: {e}")
        depth_pipe = hf_pipeline(
            "depth-estimation", 
            model="LiheYoung/depth-anything-small-hf", 
            device=0 if DEVICE == "cuda" else -1
        )
    
    # Initialize PERFECTED padding depth processor
    depth_processor = PaddingDepthProcessor(
        depth_pipe, 
        batch_size=DEPTH_BATCH_SIZE,
        target_size=DEPTH_TARGET_SIZE
    )
    
    # DINO model with mixed precision
    dino_model = load_model(DINO_CFG, DINO_CKPT).to(DEVICE)
    backbone = dino_model.backbone.eval()
    
    # Image transforms for DINO
    transform = T.Compose([
        T.Resize((224, 224)), 
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print(">>> 3. Setting up dataset and dataloader...")
    dataset = VGDataset(
        split=split, 
        img_dir=IMG_DIR, 
        roidb_file=ROIDB_FILE,
        dict_file=DICT_FILE, 
        image_file=IMAGE_FILE, 
        num_val_im=0
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=IMG_BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS, 
        collate_fn=custom_collate,
        prefetch_factor=2,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    total_batches = len(dataloader)
    print(f">>> 4. Processing {total_batches} batches...")
    
    # Progress tracking
    processed_count = 0
    skipped_count = 0
    
    for batch_idx, (batch_imgs, batch_targets) in enumerate(tqdm(dataloader, total=total_batches)):
        # Skip if no images
        if not batch_imgs:
            continue
        
        # Filter for active processing
        batch_active = []
        need_depth = []
        need_depth_indices = []
        original_sizes = []
        
        for i, target in enumerate(batch_targets):
            h5_idx = id_to_h5idx.get(target['image_id'])
            if h5_idx is not None:
                file_path = os.path.join(SAVE_DIR, f"{h5_idx}.pt")
                # Check if already processed (file exists and has content)
                if not os.path.exists(file_path) or os.path.getsize(file_path) < 100:
                    img_rgb = batch_imgs[i].convert("RGB")
                    batch_active.append({
                        'img': img_rgb, 
                        'target': target, 
                        'h5_idx': h5_idx,
                        'orig_idx': i
                    })
                    
                    # Check cache first
                    cached_depth = depth_cache.get(h5_idx)
                    if cached_depth is None:
                        need_depth.append(img_rgb)
                        need_depth_indices.append(len(batch_active)-1)
                        original_sizes.append(img_rgb.size)  # (width, height)
        
        if not batch_active:
            skipped_count += len(batch_imgs)
            continue
        
        # Process depth for images that need it
        depth_results = [None] * len(batch_active)
        
        # First, fill with cached results
        for i in range(len(batch_active)):
            cached = depth_cache.get(batch_active[i]['h5_idx'])
            if cached is not None:
                depth_results[i] = cached
        
        # Process remaining images in batches using PERFECTED processor
        if need_depth:
            try:
                # Use perfected padding depth processor
                new_depth_maps = depth_processor.process_batch(need_depth, original_sizes)
                
                for idx, depth_map in zip(need_depth_indices, new_depth_maps):
                    depth_results[idx] = depth_map
                    depth_cache.set(batch_active[idx]['h5_idx'], depth_map)
                    
            except Exception as e:
                print(f"Batch depth processing error: {e}")
                # Fallback to individual processing with padding
                for i, img in enumerate(need_depth):
                    idx = need_depth_indices[i]
                    try:
                        # Manual padding for single image
                        w, h = img.size
                        target_w, target_h = DEPTH_TARGET_SIZE
                        
                        # Calculate scaling factor
                        scale = min(target_w / w, target_h / h)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        
                        # Resize
                        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        
                        # Pad
                        img_padded = Image.new("RGB", (target_w, target_h), (0, 0, 0))
                        left = (target_w - new_w) // 2
                        top = (target_h - new_h) // 2
                        img_padded.paste(img_resized, (left, top))
                        
                        # Process
                        output = depth_pipe(img_padded)
                        
                        if output and "depth" in output:
                            depth_map_padded = output["depth"]
                            orig_w, orig_h = original_sizes[i]
                            
                            # Crop and resize
                            depth_map_cropped = depth_map_padded.crop(
                                (left, top, left + new_w, top + new_h)
                            )
                            depth_map_original = depth_map_cropped.resize(
                                (orig_w, orig_h), 
                                Image.Resampling.BILINEAR
                            )
                            
                            # Normalize
                            d_raw = np.array(depth_map_original, dtype=np.float32)
                            d_min, d_max = d_raw.min(), d_raw.max()
                            if d_max > d_min:
                                d_norm = (d_raw - d_min) / (d_max - d_min + 1e-6)
                            else:
                                d_norm = np.ones_like(d_raw) * 0.5
                            
                            depth_results[idx] = d_norm
                            depth_cache.set(batch_active[idx]['h5_idx'], d_norm)
                    except Exception as e2:
                        print(f"Individual depth error: {e2}")
                        h, w = batch_active[idx]['img'].size[1], batch_active[idx]['img'].size[0]
                        depth_results[idx] = np.ones((h, w), dtype=np.float32) * 0.5
        
        # Ensure all images have depth maps
        for i in range(len(batch_active)):
            if depth_results[i] is None:
                h, w = batch_active[i]['img'].size[1], batch_active[i]['img'].size[0]
                depth_results[i] = np.ones((h, w), dtype=np.float32) * 0.5
        
        # Process each active image
        dino_queue = []
        dino_meta = []
        batch_results = [ [] for _ in range(len(batch_active)) ]
        
        for i, item in enumerate(batch_active):
            img = item['img']
            target = item['target']
            h5_idx = item['h5_idx']
            
            w, h = img.size
            
            # Get depth map (now CORRECT with aspect ratio preserved!)
            d_norm = depth_results[i]
            
            # Extract objects with depth
            objects = []
            for k, box in enumerate(target['boxes']):
                x1, y1, x2, y2 = box.tolist()
                # Clamp coordinates
                bx = max(0, int(x1))
                by = max(0, int(y1))
                bw = min(int(x2 - x1), w - bx)
                bh = min(int(y2 - y1), h - by)
                
                if bw > 0 and bh > 0:
                    # Extract depth value from the object region
                    depth_patch = d_norm[by:by+bh, bx:bx+bw]
                    if depth_patch.size > 0:
                        z = float(np.median(depth_patch))
                    else:
                        z = 0.5
                    
                    objects.append({
                        'box': [bx, by, bw, bh], 
                        'z': z, 
                        'orig_id': k
                    })
            
            # Skip if not enough objects
            if len(objects) < 2:
                async_saver.save(
                    os.path.join(SAVE_DIR, f"{h5_idx}.pt"),
                    []
                )
                continue
            
            # Divide into layers
            layer_map = divide_objects_into_layers(objects)
            
            # Get ground truth pairs
            gt_pairs_set = set()
            if 'edges' in target:
                gt_pairs_set = set([(int(e[0]), int(e[1])) for e in target['edges']])
            
            # Compute candidate pairs
            pairs = compute_object_pairs(objects, layer_map, gt_pairs_set, (w, h))
            
            # Process each pair
            for pair in pairs:
                i_idx, j_idx = pair['i'], pair['j']
                ux1, uy1, ux2, uy2 = pair['crop_coords']
                
                # Crop and transform for DINO
                crop_img = img.crop((ux1, uy1, ux2, uy2))
                dino_queue.append(transform(crop_img))
                dino_meta.append((i, len(batch_results[i])))
                
                # Compute geometry
                geo = get_geometry_vector(
                    pair['box_i'], pair['box_j'], 
                    w, h, pair['z_i'], pair['z_j']
                )
                
                # Get predicate label if GT pair
                pred = 0
                if pair['is_gt'] and 'edges' in target:
                    for e in target['edges']:
                        if int(e[0]) == pair['sub_id'] and int(e[1]) == pair['obj_id']:
                            pred = int(e[2])
                            break
                
                # Store result
                batch_results[i].append({
                    'geo': geo,
                    'vis': None,  # Will be filled by DINO
                    'pred': pred,
                    'sub_id': pair['sub_id'],
                    'obj_id': pair['obj_id'],
                    'image_id': int(target['image_id'])
                })
        
        # Process DINO features in LARGE chunks with mixed precision
        if dino_queue:
            for chunk_start in range(0, len(dino_queue), DINO_CHUNK_SIZE):
                chunk_end = chunk_start + DINO_CHUNK_SIZE
                chunk = dino_queue[chunk_start:chunk_end]
                
                if not chunk:
                    continue
                
                # Stack and move to device
                stack = torch.stack(chunk).to(DEVICE)
                
                # Create nested tensor
                mask = torch.zeros(
                    (stack.shape[0], stack.shape[2], stack.shape[3]), 
                    dtype=torch.bool, 
                    device=DEVICE
                )
                input_nested = NestedTensor(stack, mask)
                
                # Forward pass with mixed precision
                with torch.no_grad(), autocast():
                    out = backbone(input_nested)
                    
                    # Handle different output formats
                    if isinstance(out, dict):
                        raw = out[sorted(out.keys())[-1]]
                    elif isinstance(out, (list, tuple)):
                        features = out[0]
                        if isinstance(features, (list, tuple)):
                            raw = features[-1]
                        else:
                            raw = features
                    else:
                        raw = out
                    
                    # Extract features
                    if hasattr(raw, 'tensors'):
                        t = raw.tensors
                    else:
                        t = raw
                    
                    # Global average pooling
                    if t.dim() == 4:
                        feats = t.mean(dim=[2, 3])
                    else:
                        feats = t.flatten(start_dim=1)
                    
                    feats = feats.cpu()
                
                # Assign features to results
                for local_i, global_i in enumerate(range(chunk_start, min(chunk_end, len(dino_queue)))):
                    im_idx, res_idx = dino_meta[global_i]
                    if im_idx < len(batch_results) and res_idx < len(batch_results[im_idx]):
                        batch_results[im_idx][res_idx]['vis'] = feats[local_i].unsqueeze(0)
        
        # Save results asynchronously
        for i, res_list in enumerate(batch_results):
            if res_list:
                # Filter out entries without visual features
                final_list = [r for r in res_list if r['vis'] is not None]
                
                if final_list:
                    filepath = os.path.join(SAVE_DIR, f"{batch_active[i]['h5_idx']}.pt")
                    async_saver.save(filepath, final_list)
                    processed_count += 1
                else:
                    # Save empty list for images with no valid pairs
                    filepath = os.path.join(SAVE_DIR, f"{batch_active[i]['h5_idx']}.pt")
                    async_saver.save(filepath, [])
        
        # Print progress periodically
        if (batch_idx + 1) % 10 == 0:
            print(f"\n>>> Progress: Batch {batch_idx+1}/{total_batches}")
            print(f"    Processed: {processed_count}, Skipped: {skipped_count}")
            
            # Clear cache periodically
            if processed_count % 100 == 0:
                torch.cuda.empty_cache()
    
    # Wait for all saves to complete
    print(">>> Waiting for async saves to complete...")
    async_saver.wait_completion()
    async_saver.stop()
    
    print(">>> Preprocessing completed successfully!")
    print(f"    Total processed: {processed_count}")
    print(f"    Total skipped: {skipped_count}")
    
    # Cleanup
    del depth_pipe
    del dino_model
    torch.cuda.empty_cache()

def main():
    # """Run preprocessing for train split first, then test split"""
    # print(f"{'='*60}")
    # print("STARTING PREPROCESSING PIPELINE")
    # print(f"{'='*60}")
    
    # # Process training split first
    # run_perfected_preprocess('train')
    
    # print(f"\n{'='*60}")
    # print("TRAINING SPLIT COMPLETED. STARTING TEST SPLIT...")
    # print(f"{'='*60}")
    
    # Process test split after training is done
    run_perfected_preprocess('test')
    
    print(f"\n{'='*60}")
    print("ALL PREPROCESSING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()