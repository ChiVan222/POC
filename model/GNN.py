import torch
import torch.nn as nn 
import torch.nn.functional as F 

class GNN(nn.Module): 
    def __init__(self, sam_dim = 256, clip_visual_dim = 768, clip_text_dim = 768, hidden_dim = 512, num_layers =2, edge_extra_dim=4): 
          super().__init__()
          self.node_mlp = nn.Sequential(
               nn.Linear(sam_dim, hidden_dim),
               nn.LayerNorm(hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim)
          )
          # Edge MLP now accepts visual CLIP features concatenated with spatial extras
          self.edge_mlp = nn.Sequential(
               nn.Linear(clip_visual_dim + edge_extra_dim, hidden_dim),
               nn.LayerNorm(hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim)
          )
          encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
          self.gnn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        
          self.classifier_head = nn.Sequential(
              nn.Linear(hidden_dim, hidden_dim),
              nn.LayerNorm(hidden_dim),
              nn.ReLU(),
              nn.Linear(hidden_dim, clip_text_dim) 
          )         
          self.bg_embedding = nn.Parameter(torch.randn(1, clip_text_dim))
          # Binary interaction head: predicts whether an edge represents a real interaction
          self.interaction_head = nn.Sequential(
               nn.Linear(hidden_dim, hidden_dim),
               nn.LayerNorm(hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, 1),
               nn.Sigmoid()
          )
    def forward(self, node_feats, edge_feats, edge_indices): 
        
        nodes  = self.node_mlp(node_feats)
        edges = self.edge_mlp(edge_feats)
        subj_embeds = nodes[edge_indices[:,0]]
        obj_embeds = nodes[edge_indices[:,1]]

        triplets= subj_embeds+ obj_embeds+ edges
        triplets = triplets.unsqueeze(0)
        refined_edges = self.gnn(triplets).squeeze(0)
        visual_concepts = self.classifier_head(refined_edges)
        # interaction score per edge in [0,1]
        interaction_score = self.interaction_head(refined_edges).squeeze(-1)
        return visual_concepts, interaction_score