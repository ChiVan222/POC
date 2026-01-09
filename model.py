import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialHead(nn.Module):
    def __init__(self, in_dim=8, embed_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, embed_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)

    def forward(self, geo, text_emb):
        geo_feat = F.normalize(self.mlp(geo), dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)
        return self.logit_scale.exp() * (geo_feat @ text_emb.T), geo_feat

class GatedSemanticActionHead(nn.Module):
    def __init__(self, vis_dim=1024, geo_dim=8, embed_dim=768):
        super().__init__()
        self.retention = nn.Linear(vis_dim, embed_dim)
        self.alignment = nn.Sequential(
            nn.Linear(vis_dim + geo_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, embed_dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(vis_dim, 1),
            nn.Sigmoid()
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)

    def forward(self, vis_feats, geo, text_emb):
        f_ret = self.retention(vis_feats)
        combined_in = torch.cat([vis_feats, geo], dim=-1)
        f_ali = self.alignment(combined_in) 
        
        g = self.gate(vis_feats)
        vis_aligned = F.normalize(f_ret + g * f_ali, dim=-1)
        
        text_emb = F.normalize(text_emb, dim=-1)
        logits = self.logit_scale.exp() * (vis_aligned @ text_emb.T)
        
        return logits, vis_aligned, f_ali, f_ret, g
    
class DecoupledSemanticSGG(nn.Module):
    def __init__(self, vis_dim=1024, geo_dim=8, embed_dim=768):
        super().__init__()
        self.spatial = SpatialHead(in_dim=geo_dim, embed_dim=embed_dim)
        self.action = GatedSemanticActionHead(vis_dim=vis_dim, geo_dim=geo_dim, embed_dim=embed_dim)

    def forward(self, geo, vis_feats, spatial_text, action_text):
        s_logits, s_feat = self.spatial(geo, spatial_text)
        a_logits, vis_aligned, f_ali, f_ret, g = self.action(vis_feats, geo, action_text)
        return s_logits, a_logits, vis_aligned, f_ali, f_ret, g