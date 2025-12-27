import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import clip
import os
import numpy as np
from tqdm import tqdm  # pip install tqdm

# --- IMPORTS ---
from model.GNN import GNN
from precomputed_loader import PrecomputedVGDataset 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- CONFIG ---
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 80    
VISUAL_DIM = 768  
TEXT_DIM = 768    
SAM_DIM = 256     
CHECKPOINT_DIR = "checkpoints"
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "gnn_latest.pth")
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "gnn_best.pth")

# Full VG150 Predicate List
VG150_PREDICATES = [
    "above", "across", "against", "along", "and", "at", "attached to", "behind", 
    "belonging to", "between", "carrying", "covered in", "covering", "eating", 
    "flying in", "for", "from", "growing on", "hanging from", "has", "holding", 
    "in", "in front of", "laying on", "looking at", "lying on", "made of", 
    "mounted on", "near", "of", "on", "on back of", "over", "painted on", 
    "parked on", "part of", "playing", "riding", "says", "sitting on", 
    "standing on", "to", "under", "using", "walking in", "walking on", 
    "watching", "wearing", "wears", "with"
]

def graph_collate_fn(batch):
    """
    Merges a list of graph dictionaries into a single batch.
    """
    all_node_feats = []
    all_edge_feats = []
    all_edge_indices = []
    all_text_targets = []
    
    node_offset = 0
    
    for sample in batch:
        nodes = sample['node_feats']
        edges = sample['edge_feats']
        indices = sample['edge_indices']
        
        all_node_feats.append(nodes)
        all_edge_feats.append(edges)
        all_edge_indices.append(indices + node_offset) # Shift indices
        all_text_targets.extend(sample['text_targets'])
        
        node_offset += nodes.shape[0]

    return {
        'node_feats': torch.cat(all_node_feats, dim=0),
        'edge_feats': torch.cat(all_edge_feats, dim=0),
        'edge_indices': torch.cat(all_edge_indices, dim=0),
        'text_targets': all_text_targets
    }

class DistillationLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, device="cuda"):
        super().__init__()
        self.temperature = temperature
        self.ce_loss = torch.nn.CrossEntropyLoss()
        
        # Static Negatives
        common_rels = VG150_PREDICATES
        try:
            import clip
            print(f"Encoding {len(common_rels)} static negatives...")
            clip_model, _ = clip.load("ViT-L/14", device=device)
            with torch.no_grad():
                prompts = [f"a photo of something {r} something" for r in common_rels]
                toks = clip.tokenize(prompts).to(device)
                self.neg_embeds = clip_model.encode_text(toks).float()
                self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)
            del clip_model
            torch.cuda.empty_cache()
        except:
            self.neg_embeds = None

    def forward(self, student_embeds, teacher_text_embeds):
        student_norm = torch.nn.functional.normalize(student_embeds, dim=-1)
        teacher_norm = torch.nn.functional.normalize(teacher_text_embeds, dim=-1)
        
        if self.neg_embeds is not None:
            all_targets = torch.cat([teacher_norm, self.neg_embeds], dim=0)
        else:
            all_targets = teacher_norm
            
        logits = torch.matmul(student_norm, all_targets.T) / self.temperature
        labels = torch.arange(student_embeds.size(0)).to(student_embeds.device)
        return self.ce_loss(logits, labels)

import glob
import re

def train_fast():
    print(f"Initializing FAST Training on {DEVICE}...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 1. Initialize Student (GNN)
    student_gnn = GNN(
        sam_dim=SAM_DIM,
        clip_visual_dim=VISUAL_DIM,
        clip_text_dim=TEXT_DIM,
        hidden_dim=512
    ).to(DEVICE)
    student_gnn.train()
    
    # 2. Initialize Teacher
    print("Loading CLIP (Teacher)...")
    clip_model, _ = clip.load("ViT-L/14", device=DEVICE)
    for p in clip_model.parameters(): p.requires_grad = False

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(student_gnn.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience= 1
    )

    # --- 4. SMART RESUME LOGIC ---
    start_epoch = 0
    best_loss = float('inf')
    
    # Strategy: Find the checkpoint with the highest epoch number
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "gnn_epoch_*.pth"))
    
    # Sort files by epoch number
    def extract_epoch(f):
        match = re.search(r"gnn_epoch_(\d+).pth", f)
        return int(match.group(1)) if match else -1

    if checkpoint_files:
        latest_file = max(checkpoint_files, key=extract_epoch)
        print(f"🔄 Found checkpoint: {latest_file}")
        
        try:
            checkpoint = torch.load(latest_file)
            
            # CASE A: New Style (Full Checkpoint Dictionary)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                student_gnn.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_loss = checkpoint.get('best_loss', float('inf'))
                print(f"   -> Full Resume: Starting at Epoch {start_epoch}")
                
            # CASE B: Old Style (Weights Only - Your gnn_epoch_19.pth is likely this)
            else:
                student_gnn.load_state_dict(checkpoint)
                # Infer epoch from filename
                start_epoch = extract_epoch(latest_file) + 1
                print(f"   -> Weights-Only Resume: Starting at Epoch {start_epoch} (Optimizer reset)")
                
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            print("   -> Starting from scratch.")
    else:
        print("   -> No checkpoints found. Starting from scratch.")

    criterion = DistillationLoss(device=DEVICE)
    
    dataset = PrecomputedVGDataset(data_dir="datasets/vg/precomputed_features")
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=graph_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    print(f"🚀 Start Training on {len(dataset)} samples...")
    
    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0
        
        # Progress Bar
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        
        for batch in pbar:
            node_tensor = batch['node_feats'].to(DEVICE).float()
            edge_tensor = batch['edge_feats'].to(DEVICE).float()
            idx_tensor = batch['edge_indices'].to(DEVICE)
            text_prompts = batch['text_targets']
            
            if len(text_prompts) == 0: continue

            # Teacher
            text_tokens = clip.tokenize(text_prompts, truncate=True).to(DEVICE)
            with torch.no_grad():
                teacher_embeds = clip_model.encode_text(text_tokens).float()

            # Student
            optimizer.zero_grad()
            student_embeds = student_gnn(node_tensor, edge_tensor, idx_tensor)
            
            loss = criterion(student_embeds, teacher_embeds)
            
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            total_loss += current_loss
            
            # Update Bar
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f"{current_loss:.4f}", 'lr': f"{current_lr:.6f}"})

        # End of Epoch
        avg_loss = total_loss / len(loader)
        
        # Step Scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"📉 Scheduler: Learning Rate reduced from {old_lr} to {new_lr}")
        
        # Save FULL Checkpoint (New Style)
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': student_gnn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss
        }
        
        # Save numbered epoch file (so you have history)
        torch.save(checkpoint_data, os.path.join(CHECKPOINT_DIR, f"gnn_epoch_{epoch}.pth"))
        
        # Save Best
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_data['best_loss'] = best_loss
            torch.save(checkpoint_data, BEST_CHECKPOINT)
            print(f"🌟 New Best Model Saved! Loss: {best_loss:.4f}")
            
        print(f"=== Epoch {epoch} Finished. Avg Loss: {avg_loss:.4f} ===")

if __name__ == "__main__":
    train_fast()