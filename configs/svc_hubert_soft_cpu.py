_base_ = ["./svc_hubert_soft.py"]

trainer = dict(
    accelerator="cpu",
    devices=1,
    precision=32,
    max_steps=10_000,
)
