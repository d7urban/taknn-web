import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, channels):
        super().__init__()

    def forward(self, x, gamma, beta):
        # x: [B, C, H, W], gamma: [B, C], beta: [B, C]
        return x * gamma.view(-1, x.size(1), 1, 1) + beta.view(-1, x.size(1), 1, 1)

class ResidualBlock(nn.Module):
    def __init__(self, channels, film_embed_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.film1 = FiLM(channels)
        
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.film2 = FiLM(channels)

        self.gamma1 = nn.Linear(film_embed_dim, channels)
        self.beta1 = nn.Linear(film_embed_dim, channels)
        self.gamma2 = nn.Linear(film_embed_dim, channels)
        self.beta2 = nn.Linear(film_embed_dim, channels)

    def forward(self, x, e):
        residual = x
        
        g1 = self.gamma1(e)
        b1 = self.beta1(e)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.film1(out, g1, b1)
        out = F.relu(out)
        
        g2 = self.gamma2(e)
        b2 = self.beta2(e)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.film2(out, g2, b2)
        
        out += residual
        return F.relu(out)

class TeacherModel(nn.Module):
    def __init__(self, channels=128, num_blocks=10, film_embed_dim=32):
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
        # input: g (128) + h_src (128) + h_dst (128) + path_pool (128) + discrete (64) + flags (3) = 579
        self.policy_mlp = nn.Sequential(
            nn.Linear(channels * 4 + 64 + 3, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Discrete embeds for policy
        self.move_type_emb = nn.Embedding(2, 8)
        self.piece_type_emb = nn.Embedding(4, 8)
        self.direction_emb = nn.Embedding(5, 8)
        self.pickup_count_emb = nn.Embedding(9, 16)
        self.drop_template_emb = nn.Embedding(256, 16)
        self.travel_length_emb = nn.Embedding(8, 8)

        # Value Head
        self.v_hidden = nn.Linear(channels + film_embed_dim, 128)
        self.wdl_head = nn.Linear(128, 3)
        self.margin_head = nn.Linear(128, 1)

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

    def score_moves(self, trunk_outputs, move_descriptors):
        # move_descriptors should be a batch of move features.
        # This part is complex because of variable legal moves.
        # Usually implemented as a separate step or with padding.
        pass
