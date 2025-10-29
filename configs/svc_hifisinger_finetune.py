from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

_base_ = [
    "./_base_/archs/hifi_svc.py",
    "./_base_/trainers/base.py",
    "./_base_/schedulers/exponential.py",
    "./_base_/datasets/hifi_svc.py",
]

speaker_mapping = {
    "placeholder": 0,
}

model = dict(
    type="HiFiSVC",
    speaker_encoder=dict(
        input_size=len(speaker_mapping),
    ),
)

preprocessing = dict(
    text_features_extractor=dict(
        type="ContentVec",
    ),
    pitch_extractor=dict(
        type="ParselMouthPitchExtractor",
        keep_zeros=False,
        f0_min=40.0,
        f0_max=1600.0,
    ),
    energy_extractor=dict(
        type="RMSEnergyExtractor",
    ),
    augmentations=[
        dict(
            type="FixedPitchShifting",
            key_shifts=[-5.0, 5.0],
            probability=1.5,
        ),
    ],
)

trainer = dict(
    # Disable gradient clipping, which is not supported by custom optimization
    gradient_clip_val=None,
    val_check_interval=500,  # MARGOT: Validate every 500 steps for better checkpoint selection
    check_val_every_n_epoch=None,
    max_steps=20000,  # MARGOT: Optimal for 30min datasets (user configurable in notebook)
    callbacks=[
        ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.4f}",  # MARGOT: 4 decimals for proper sorting (e.g., 0.9420)
            every_n_train_steps=500,  # MARGOT: Save every 500 steps (2x granularity)
            save_top_k=10,  # MARGOT: Keep best 10 checkpoints by validation loss
            monitor="valid_loss_epoch",  # MARGOT: Sort by validation loss (epoch-level metric)
            mode="min",  # MARGOT: Lower validation loss is better
        ),
        LearningRateMonitor(logging_interval="step"),
    ],
)
