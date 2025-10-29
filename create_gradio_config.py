#!/usr/bin/env python3
"""
Create a gradio-config.yaml file for running the Gradio UI locally.
Prompts for checkpoint path and config path.
"""

import yaml
import os
import glob

def find_checkpoints():
    """Find checkpoint files in common locations."""
    patterns = [
        "logs/HiFiSVC/*/checkpoints/*.ckpt",
        "checkpoints/*.ckpt",
        "*.ckpt",
    ]

    checkpoints = []
    for pattern in patterns:
        checkpoints.extend(glob.glob(pattern))

    return sorted(checkpoints)

def main():
    print("=" * 70)
    print("MARGOT - Create Gradio Config")
    print("=" * 70)

    # Find checkpoints
    checkpoints = find_checkpoints()

    if checkpoints:
        print("\n📦 Found checkpoints:")
        for i, ckpt in enumerate(checkpoints, 1):
            print(f"  {i}. {ckpt}")

        choice = input(f"\nSelect checkpoint (1-{len(checkpoints)}) or enter custom path: ").strip()

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                checkpoint_path = checkpoints[idx]
            else:
                checkpoint_path = choice
        except ValueError:
            checkpoint_path = choice
    else:
        checkpoint_path = input("\n📦 Enter checkpoint path (file or folder): ").strip()

    # Config path
    config_path = input("\n⚙️  Enter config path [configs/svc_hifisinger_finetune.py]: ").strip()
    if not config_path:
        config_path = "configs/svc_hifisinger_finetune.py"

    # Model name
    model_name = input("\n🎤 Enter model name [My MARGOT Model]: ").strip()
    if not model_name:
        model_name = "My MARGOT Model"

    # Create the config
    gradio_config = {
        "readme": "# MARGOT - HiFiSinger Demo 🎤\nGitHub: [gdoscher/MARGOT](https://github.com/gdoscher/MARGOT)\n",
        "max_mixing_speakers": 3,
        "models": [
            {
                "name": model_name,
                "config": config_path,
                "checkpoint": checkpoint_path,
                "readme": "This model is pretrained on Opencpop and M4Singer and finetuned on your dataset.",
            }
        ]
    }

    # Write the config
    output_file = "gradio-config.yaml"
    with open(output_file, "w") as f:
        yaml.dump(gradio_config, f, default_flow_style=False)

    print("\n" + "=" * 70)
    print("✅ Config created successfully!")
    print("=" * 70)
    print(f"📁 Saved to: {output_file}")
    print(f"\n📝 Config contents:")
    print(f"  Model name:   {model_name}")
    print(f"  Config:       {config_path}")
    print(f"  Checkpoint:   {checkpoint_path}")
    print("\n🚀 To run Gradio UI:")
    print(f"  python tools/hifisinger/gradio_ui.py --config {output_file}")
    print("\n💡 Or with sharing enabled:")
    print(f"  python tools/hifisinger/gradio_ui.py --config {output_file} --share")

if __name__ == "__main__":
    main()
