import torch
import torch.nn as nn
from models.utils import _init_weights

class FusionAttention(nn.Module):
    def __init__(self, num_heads=8, hidden_dim=256, dropout=0.1):
        super(FusionAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden_dim)
        
        self.layernorm.apply(_init_weights)
    
    def forward(self, query, key, value):
        hidden_states = self.attention(query, key, value)[0]
        outputs = self.layernorm(hidden_states + query)
        return outputs

class FusionFFN(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super(FusionFFN, self).__init__()

        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.apply(_init_weights)
    
    def forward(self, inputs):
        hidden_f = self.linear(inputs)
        hidden_f = self.dropout(hidden_f)
        outputs = self.layernorm(hidden_f + inputs)
        return outputs

class FusionBlock(nn.Module):
    def __init__(self, 
            num_heads=8, 
            hidden_dim=256, 
            dropout=0.1
        ):
        super(FusionBlock, self).__init__()

        # Cross-attention layers
        self.ca_scene_mot = FusionAttention(num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout)
        self.ca_mot_scene = FusionAttention(num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout)
        
        # Feed Forward Networks
        self.ffn_scene = FusionFFN(hidden_dim=hidden_dim, dropout=dropout)
        self.ffn_motion = FusionFFN(hidden_dim=hidden_dim, dropout=dropout)
        
        ## weight initialization
    
    def forward(self, scene_f, motion_f):
        # Cross-Attention: motion_f attends to scene_f
        scene_cross_f = self.ca_scene_mot(
            query=scene_f, 
            key=motion_f, 
            value=motion_f
        )
        
        # Cross-Attention: scene_f attends to motion_f
        motion_cross_f = self.ca_mot_scene(
            query=motion_f, 
            key=scene_f, 
            value=scene_f
        )
        
        # Feed Forward Block
        scene_output_f = self.ffn_scene(scene_cross_f)
        motion_output_f = self.ffn_motion(motion_cross_f)
          
        return scene_output_f, motion_output_f

class SMFusion(nn.Module):
    def __init__(self, hidden_dim, num_blocks, num_heads=8, dropout=0.1):
        super(SMFusion, self).__init__()
        self.blocks = nn.ModuleList([
            FusionBlock(num_heads, hidden_dim, dropout) for _ in range(num_blocks)
        ])
    
    def forward(self, scene_f, motion_f):
        for block in self.blocks:
            scene_f, motion_f = block(scene_f, motion_f) # [B, L, d]
        scene_f = torch.nn.functional.normalize(scene_f, dim=-1)
        motion_f = torch.nn.functional.normalize(motion_f, dim=-1)
        return scene_f, motion_f

if __name__ == "__main__":
    # Example usage
    batch_size = 32
    N1, N2 = 128, 256
    d = 512
    num_blocks = 4
    num_heads = 8
    ff_dim = 2048

    scene_f = torch.rand(batch_size, N1, d)
    motion_f = torch.rand(batch_size, N2, d)

    fusion_model = SMFusion(d=d, num_blocks=num_blocks, num_heads=num_heads, ff_dim=ff_dim)
    A_fused, B_fused = fusion_model(scene_f, motion_f)