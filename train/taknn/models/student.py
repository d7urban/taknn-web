import torch
import torch.nn as nn
import torch.nn.functional as F
from .teacher import ResidualBlock, PolicyScorerMixin

class StudentModel(PolicyScorerMixin, nn.Module):
    def __init__(self, channels=64, num_blocks=6, film_embed_dim=16):
        super().__init__()
        self.size_embed = nn.Embedding(6, film_embed_dim)
        
        self.stem = nn.Sequential(
            nn.Conv2d(31, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, film_embed_dim) for _ in range(num_blocks)
        ])
        
        # Policy Head MLP
        # input: g (64) + h_src (64) + h_dst (64) + path_pool (64) + discrete (64) + flags (3) = 323
        self.policy_mlp = nn.Sequential(
            nn.Linear(channels * 4 + 64 + 3, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Discrete embeds for policy
        self.move_type_emb = nn.Embedding(2, 8)
        self.piece_type_emb = nn.Embedding(4, 8)
        self.direction_emb = nn.Embedding(5, 8)
        self.pickup_count_emb = nn.Embedding(9, 16)
        self.drop_template_emb = nn.Embedding(256, 16)
        self.travel_length_emb = nn.Embedding(8, 8)

        # Value Head
        self.v_hidden = nn.Linear(channels + film_embed_dim, 64)
        self.wdl_head = nn.Linear(64, 3)
        self.margin_head = nn.Linear(64, 1)

        # Aux Heads
        self.road_threat = nn.Conv2d(channels, 2, 1)
        self.block_threat = nn.Conv2d(channels, 2, 1)
        self.cap_flatten = nn.Conv2d(channels, 1, 1)
        self.endgame_head = nn.Linear(channels + film_embed_dim, 1)

    def forward(self, board_tensor, size_id):
        # board_tensor: [B, 31, 8, 8], size_id: [B]
        e = self.size_embed(size_id)
        
        x = self.stem(board_tensor)
        for block in self.blocks:
            x = block(x, e)
            
        h = x # spatial trunk output [B, C, 8, 8]
        g = F.adaptive_avg_pool2d(h, (1, 1)).view(h.size(0), -1) # global pool [B, C]
        
        # Value & Aux
        v_in = torch.cat([g, e], dim=1)
        v_hid = F.relu(self.v_hidden(v_in))
        wdl = F.softmax(self.wdl_head(v_hid), dim=1)
        margin = torch.tanh(self.margin_head(v_hid))
        
        road = torch.sigmoid(self.road_threat(h))
        block = torch.sigmoid(self.block_threat(h))
        cap = torch.sigmoid(self.cap_flatten(h))
        endgame = torch.sigmoid(self.endgame_head(v_in))
        
        return {
            "spatial": h,
            "global": g,
            "wdl": wdl,
            "margin": margin,
            "road": road,
            "block": block,
            "cap": cap,
            "endgame": endgame
        }
