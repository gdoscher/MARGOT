"""
M4 Mac with MPS (GPU) - Test configuration.

This attempts to use Apple Silicon GPU for much faster training.
If it works: 3-10x speedup!
If it crashes: Fall back to CPU version.
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

# Override trainer settings for M4 Mac with MPS (GPU)
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

trainer = dict(
    accelerator="mps",  # Use Apple Silicon GPU!
    devices=1,
    gradient_clip_val=0.5,
    gradient_clip_algorithm="norm",
    log_every_n_steps=1,
    val_check_interval=100,
    check_val_every_n_epoch=None,
    max_steps=2100,  # Test 100 more steps (from 2000 to 2100)
    precision=32,  # MPS doesn't support bf16-mixed well
    accumulate_grad_batches=1,
    callbacks=[
        ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.4f}",
            save_on_train_epoch_end=False,
            save_top_k=5,
            monitor="valid_loss",
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
    ],
)

# Override dataloader settings for M4
dataloader = dict(
    train=dict(
        batch_size=16,  # Same as CPU config
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=False,  # Not needed for MPS
    ),
    valid=dict(
        batch_size=8,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        pin_memory=False,
    ),
)
