dataset = dict(
    train=dict(
        type="NaiveSVCDataset",
        path="dataset/train",
        speaker_id=0,
    ),
    valid=dict(
        type="NaiveSVCDataset",
        path="dataset/valid",
        speaker_id=0,
    ),
)

dataloader = dict(
    train=dict(
        batch_size=20,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid shared memory issues in Docker
        persistent_workers=False,  # Must be False when num_workers=0
        pin_memory=False,  # Not needed for CPU/MPS training
    ),
    valid=dict(
        batch_size=2,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid shared memory issues in Docker
        persistent_workers=False,  # Must be False when num_workers=0
        pin_memory=False,  # Not needed for CPU/MPS training
    ),
)
