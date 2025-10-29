"""
M4 Mac-optimized configuration for Fish-Diffusion training.

This config is specifically designed for training on Apple M4 chip (or other Apple Silicon).
Key differences from standard config:
- CPU accelerator (MPS has compatibility issues with this codebase)
- FP32 precision
- RMVPE pitch extractor (doesn't require parselmouth - pure PyTorch)
- Optimized DataLoader settings
- Adjusted validation and logging intervals for monitoring
"""

_base_ = [
    "./_base_/archs/diff_svc_v2.py",
    "./_base_/trainers/base.py",
    "./_base_/schedulers/warmup_cosine.py",
    "./_base_/datasets/naive_svc.py",
]

# Use RMVPE instead of ParselMouth (no compilation needed!)
preprocessing = dict(
    text_features_extractor=dict(
        type="HubertSoft",
    ),
    pitch_extractor=dict(
        type="RMVPitchExtractor",  # Pure PyTorch, works in any venv
        hop_length=512,
        f0_min=50.0,
        f0_max=1100.0,
        keep_zeros=False,
        threshold=0.03,  # Confidence threshold
    ),
)

# Override trainer settings for M4 Mac
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

trainer = dict(
    accelerator="cpu",  # Use CPU - MPS has issues with this version of PyTorch Lightning
    devices=1,
    gradient_clip_val=0.5,
    gradient_clip_algorithm="norm",
    log_every_n_steps=1,  # Log every step for better monitoring with small datasets
    val_check_interval=100,  # Validate every 100 steps instead of 5000 for faster feedback
    check_val_every_n_epoch=None,
    max_steps=5_000,  # Target for CPU training (~27 hours from step 2000)
    precision=32,  # Use FP32
    accumulate_grad_batches=1,
    callbacks=[
        ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.4f}",
            save_on_train_epoch_end=False,
            save_top_k=5,  # Keep only best 5 checkpoints (saves ~65GB disk space!)
            monitor="valid_loss",  # Monitor validation loss
            mode="min",  # Keep checkpoints with lowest loss
        ),
        LearningRateMonitor(logging_interval="step"),
    ],
)

# Override dataloader settings for M4
dataloader = dict(
    train=dict(
        batch_size=16,  # Increased for more stable gradients
        shuffle=True,
        num_workers=4,  # Use 4 workers for faster data loading locally
        persistent_workers=True,
        pin_memory=False,  # Not needed for CPU
    ),
    valid=dict(
        batch_size=8,  # Increased to match
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        pin_memory=False,
    ),
)
