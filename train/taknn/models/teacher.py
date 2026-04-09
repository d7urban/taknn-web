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

class PolicyScorerMixin:
    """Shared score_moves() logic for teacher and student models."""

    def score_moves(self, trunk_outputs, descriptors, num_moves):
        """Score legal moves using the policy MLP.

        Args:
            trunk_outputs: dict with 'spatial' [B, C, 8, 8] and 'global' [B, C]
            descriptors: dict of padded tensors:
                src, dst: [B, M] square indices (0-63)
                path: [B, M, 7] square indices (255 = padding)
                move_type, piece_type, direction, pickup_count,
                drop_template_id, travel_length: [B, M] ints
                capstone_flatten, enters_occupied, opening_phase: [B, M] floats
            num_moves: [B] int tensor, actual move count per sample

        Returns:
            logits: [B, M] with -inf at padding positions
        """
        h = trunk_outputs["spatial"]  # [B, C, 8, 8]
        g = trunk_outputs["global"]   # [B, C]
        B, C, _, _ = h.shape
        M = descriptors["src"].shape[1]

        # Flatten spatial to [B, 64, C] for gathering
        h_flat = h.reshape(B, C, 64).permute(0, 2, 1)  # [B, 64, C]

        # Gather h_src and h_dst: [B, M, C]
        src_idx = descriptors["src"].long().clamp(0, 63)
        dst_idx = descriptors["dst"].long().clamp(0, 63)
        h_src = torch.gather(h_flat, 1, src_idx.unsqueeze(-1).expand(-1, -1, C))
        h_dst = torch.gather(h_flat, 1, dst_idx.unsqueeze(-1).expand(-1, -1, C))

        # Path pooling: mean of h at path squares, masked
        path = descriptors["path"].long().clamp(0, 63)  # [B, M, 7]
        path_mask = descriptors["path"] != 255  # [B, M, 7]
        path_flat = path.reshape(B, M * 7)
        h_path_flat = torch.gather(h_flat, 1, path_flat.unsqueeze(-1).expand(-1, -1, C))
        h_path = h_path_flat.reshape(B, M, 7, C)
        path_mask_f = path_mask.unsqueeze(-1).float()
        path_sum = (h_path * path_mask_f).sum(dim=2)
        path_count = path_mask_f.sum(dim=2).clamp(min=1.0)
        path_pool = path_sum / path_count  # [B, M, C]

        # Discrete embeddings
        e_move_type = self.move_type_emb(descriptors["move_type"].long())
        e_piece_type = self.piece_type_emb(descriptors["piece_type"].long())
        e_direction = self.direction_emb(descriptors["direction"].long())
        e_pickup = self.pickup_count_emb(descriptors["pickup_count"].long())
        e_template = self.drop_template_emb(descriptors["drop_template_id"].long())
        e_travel = self.travel_length_emb(descriptors["travel_length"].long())
        discrete = torch.cat([e_move_type, e_piece_type, e_direction,
                              e_pickup, e_template, e_travel], dim=-1)  # [B, M, 64]

        # Flags
        flags = torch.stack([
            descriptors["capstone_flatten"],
            descriptors["enters_occupied"],
            descriptors["opening_phase"],
        ], dim=-1)  # [B, M, 3]

        # Expand global pool
        g_exp = g.unsqueeze(1).expand(-1, M, -1)

        # Concatenate and run MLP
        mlp_input = torch.cat([g_exp, h_src, h_dst, path_pool, discrete, flags], dim=-1)
        logits = self.policy_mlp(mlp_input).squeeze(-1)  # [B, M]

        # Mask padding
        move_mask = torch.arange(M, device=logits.device).unsqueeze(0) < num_moves.unsqueeze(1)
        logits = logits.masked_fill(~move_mask, float("-inf"))

        return logits


class TeacherModel(PolicyScorerMixin, nn.Module):
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

