"""Training configuration for TakNN models."""

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # Model
    channels: int = 128
    num_blocks: int = 10
    film_embed_dim: int = 32
    model_type: str = "teacher"  # "teacher" or "student"

    # Optimizer
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    batch_size: int = 1024
    gradient_clip: float = 1.0
    mixed_precision: bool = True

    # Loss weights
    w_policy: float = 1.0
    w_wdl: float = 1.0
    w_margin: float = 0.5
    w_road: float = 0.2
    w_block: float = 0.2
    w_cap: float = 0.1
    w_endgame: float = 0.1

    # Training
    epochs: int = 50
    val_split: float = 0.1
    eval_every: int = 1
    checkpoint_dir: str = "checkpoints"
    log_every: int = 10  # batches

    # Data
    shard_dir: str = "shards"
    size_quotas: dict = field(default_factory=lambda: {
        3: 0.02, 4: 0.10, 5: 0.25, 6: 0.55, 7: 0.05, 8: 0.03,
    })
