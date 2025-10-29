#!/usr/bin/env python3
"""
Interactive inference script for HiFiSinger voice conversion.
Prompts user for all required parameters.
"""

import os
import glob
import torch
from pathlib import Path
from inference import HiFiSingerSVCInference
from mmengine import Config


def find_checkpoints(directory=None):
    """Find all checkpoint files in common locations."""
    search_paths = [
        "logs/HiFiSVC/*/checkpoints/*.ckpt",
        "checkpoints/*.ckpt",
        "../../../checkpoints/*.ckpt",
    ]

    if directory:
        search_paths.insert(0, os.path.join(directory, "*.ckpt"))

    checkpoints = []
    for pattern in search_paths:
        checkpoints.extend(glob.glob(pattern))

    return sorted(checkpoints)


def find_configs(directory=None):
    """Find all config files in common locations."""
    search_paths = [
        "configs/svc_*.py",
        "../../configs/svc_*.py",
    ]

    if directory:
        search_paths.insert(0, os.path.join(directory, "*.py"))

    configs = []
    for pattern in search_paths:
        configs.extend(glob.glob(pattern))

    return sorted(configs)


def prompt_with_default(prompt, default=None, type_converter=str):
    """Prompt user with optional default value."""
    if default is not None:
        prompt_text = f"{prompt} [{default}]: "
    else:
        prompt_text = f"{prompt}: "

    value = input(prompt_text).strip()

    if not value and default is not None:
        return type_converter(default)
    elif value:
        return type_converter(value)
    else:
        return None


def select_from_list(items, item_type="item"):
    """Let user select from a list of items."""
    if not items:
        print(f"❌ No {item_type}s found!")
        return None

    print(f"\nAvailable {item_type}s:")
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item}")

    while True:
        choice = input(f"\nSelect {item_type} number (or press Enter for #1): ").strip()

        if not choice:
            return items[0]

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                return items[idx]
            else:
                print(f"Please enter a number between 1 and {len(items)}")
        except ValueError:
            print("Please enter a valid number")


def main():
    print("=" * 70)
    print("MARGOT - Interactive HiFiSinger Inference")
    print("=" * 70)

    # 1. Select config file
    print("\n📁 Step 1: Select Configuration File")
    configs = find_configs()
    config_path = select_from_list(configs, "config")

    if not config_path:
        config_path = input("Enter path to config file: ").strip()
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return

    print(f"✅ Using config: {config_path}")

    # 2. Select checkpoint
    print("\n💾 Step 2: Select Checkpoint")

    # Try to find checkpoints near the config
    checkpoint_dir = os.path.dirname(os.path.dirname(config_path))
    checkpoints = find_checkpoints(checkpoint_dir)

    if checkpoints:
        # Sort by validation loss (extract from filename)
        def get_loss(ckpt):
            import re
            match = re.search(r'valid_loss=([\d.]+)', ckpt)
            if match:
                return float(match.group(1).rstrip('.'))
            return 999.0

        checkpoints_sorted = sorted(checkpoints, key=get_loss)

        print("\n🏆 Checkpoints sorted by validation loss (best first):")
        for i, ckpt in enumerate(checkpoints_sorted[:10], 1):  # Show top 10
            filename = os.path.basename(ckpt)
            loss_match = re.search(r'valid_loss=([\d.]+)', filename)
            loss = loss_match.group(1) if loss_match else "unknown"
            print(f"  {i}. {filename} (loss: {loss})")

        checkpoint_path = select_from_list(checkpoints_sorted, "checkpoint")
    else:
        print("No checkpoints found in common locations.")
        checkpoint_path = input("Enter path to checkpoint file: ").strip()

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        return

    print(f"✅ Using checkpoint: {checkpoint_path}")

    # 3. Input audio file
    print("\n🎵 Step 3: Input Audio File")
    input_path = input("Enter path to input audio file (.wav, .mp3, etc.): ").strip()

    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return

    print(f"✅ Input: {input_path}")

    # 4. Output path
    print("\n💿 Step 4: Output Audio File")
    default_output = os.path.splitext(input_path)[0] + "_converted.wav"
    output_path = prompt_with_default("Enter output path", default_output)

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"✅ Output: {output_path}")

    # 5. Speaker (usually 0 for single-speaker models)
    print("\n👤 Step 5: Speaker Settings")
    speaker = prompt_with_default("Speaker ID", "0")

    # 6. Pitch adjustment
    print("\n🎹 Step 6: Pitch Adjustment")
    print("  (Semitones: +12 = one octave up, -12 = one octave down)")
    pitch_adjust = prompt_with_default("Pitch adjustment", "0", int)

    # 7. Device selection
    print("\n🖥️  Step 7: Device Selection")
    if torch.cuda.is_available():
        print("  GPU detected!")
        device = prompt_with_default("Device (cuda/cpu/mps)", "cuda")
    elif torch.backends.mps.is_available():
        print("  Apple Silicon detected!")
        device = prompt_with_default("Device (mps/cpu)", "mps")
    else:
        print("  Using CPU (this will be slow)")
        device = "cpu"

    # 8. Advanced options
    print("\n⚙️  Step 8: Advanced Options (Optional)")
    extract_vocals = prompt_with_default("Extract vocals first? (y/n)", "n").lower() == 'y'

    silence_threshold = 60
    max_slice_duration = 30
    if extract_vocals:
        silence_threshold = prompt_with_default("Silence threshold (dB)", "60", int)
        max_slice_duration = prompt_with_default("Max slice duration (sec)", "30", int)

    # Summary
    print("\n" + "=" * 70)
    print("INFERENCE SETTINGS SUMMARY")
    print("=" * 70)
    print(f"Config:           {config_path}")
    print(f"Checkpoint:       {os.path.basename(checkpoint_path)}")
    print(f"Input:            {input_path}")
    print(f"Output:           {output_path}")
    print(f"Speaker:          {speaker}")
    print(f"Pitch Adjust:     {pitch_adjust:+d} semitones")
    print(f"Device:           {device}")
    print(f"Extract Vocals:   {extract_vocals}")
    print("=" * 70)

    proceed = input("\n▶️  Proceed with inference? (y/n) [y]: ").strip().lower()
    if proceed and proceed != 'y':
        print("Cancelled.")
        return

    # Run inference
    print("\n🚀 Starting inference...")

    try:
        device_obj = torch.device(device)
        config = Config.fromfile(config_path)

        print("📦 Loading model...")
        model = HiFiSingerSVCInference(config, checkpoint_path)
        model = model.to(device_obj)

        print("🎤 Processing audio...")
        model.inference(
            input_path=input_path,
            output_path=output_path,
            speaker=speaker,
            pitch_adjust=pitch_adjust,
            extract_vocals=extract_vocals,
            silence_threshold=silence_threshold,
            max_slice_duration=max_slice_duration,
        )

        print("\n" + "=" * 70)
        print("✅ INFERENCE COMPLETE!")
        print("=" * 70)
        print(f"📁 Output saved to: {output_path}")

        # Show file size
        output_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"📊 File size: {output_size:.2f} MB")

    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import re  # Import at module level for checkpoint sorting
    main()
