"""
Docker-optimized config using Crepe pitch extractor instead of ParselMouth.
Configured for CPU training inside Docker containers.
"""

_base_ = [
    "./_base_/archs/diff_svc_v2.py",
    "./_base_/trainers/base.py",
    "./_base_/schedulers/warmup_cosine.py",
    "./_base_/datasets/naive_svc.py",
]

preprocessing = dict(
    text_features_extractor=dict(
        type="HubertSoft",
    ),
    pitch_extractor=dict(
        type="CrepePitchExtractor",  # Use Crepe instead of ParselMouth
        # Crepe is slower but more accurate and doesn't require parselmouth
        keep_zeros=False,
        hop_length=512,
        f0_min=50.0,
        f0_max=1100.0,
        model="tiny",  # Options: tiny, small, medium, large, full
    ),
)

# Override trainer settings for Docker (CPU-only)
trainer = dict(
    accelerator="cpu",  # Docker containers can't access M4 GPU
    devices=1,
    gradient_clip_val=0.5,
    gradient_clip_algorithm="norm",
    log_every_n_steps=1,  # Log every step for small datasets
    val_check_interval=100,  # Validate every 100 steps
    check_val_every_n_epoch=None,
    max_steps=100_000,
    precision=32,  # FP32 for CPU training
    accumulate_grad_batches=1,
)

# DataLoader settings already configured in base config for Docker
# (num_workers=0, no persistent workers, no pin_memory)
