import sys

import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default

trainer = dict(
    accelerator="gpu",
    devices=-1,
    gradient_clip_val=0.5,
    gradient_clip_algorithm="norm",
    log_every_n_steps=1,  # Changed from 10 to 1 to support small datasets
    val_check_interval=500,  # MARGOT: Increased granularity for better checkpoint selection
    check_val_every_n_epoch=None,
    max_steps=20000,  # MARGOT: Optimal for pretrained fine-tuning (user configurable in notebook)
    # Warning: If you are training the model with fs2 (and see nan), you should either use bf16 or fp32
    precision="16-mixed",  # MARGOT: Changed from bf16 to fp16 for T4 GPU compatibility
    accumulate_grad_batches=1,
    callbacks=[
        ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.2f}",
            save_on_train_epoch_end=False,
            every_n_train_steps=500,  # MARGOT: Save every 500 steps for better granularity
            save_top_k=10,  # MARGOT: Keep best 10 checkpoints by validation loss
            monitor="valid_loss",  # MARGOT: Sort by validation loss
            mode="min",  # MARGOT: Lower validation loss is better
        ),
        LearningRateMonitor(logging_interval="step"),
    ],
)

# Use DDP for multi-gpu training
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    # Use gloo for windows
    process_group_backend = "nccl" if sys.platform != "win32" else "gloo"

    trainer["strategy"] = DDPStrategy(
        process_group_backend=process_group_backend,
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
        static_graph=True,
        ddp_comm_hook=default.fp16_compress_hook,
    )
