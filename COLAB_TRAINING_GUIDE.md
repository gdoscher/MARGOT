# Google Colab Training Guide - HiFiSinger

Complete step-by-step guide for training voice conversion models using Google Colab's free GPU.

---

## 📋 Prerequisites

- [ ] Google account
- [ ] 30+ minutes of clean vocal audio (single speaker)
- [ ] ~2GB Google Drive space for dataset + checkpoints

---

## 🎯 Step 1: Prepare Your Dataset

### Dataset Requirements

**Audio Quality:**
- ✅ Single speaker only
- ✅ Clean vocals (no background music/instruments)
- ✅ WAV format, 44.1kHz sampling rate
- ✅ Total duration: 30-60 minutes recommended
- ✅ Individual files: Any length (Colab will handle splitting)

**What to Avoid:**
- ❌ Multiple speakers
- ❌ Background music or noise
- ❌ MP3 or other compressed formats
- ❌ Low-quality recordings

### Creating Your Dataset ZIP

**Option 1: Pre-split (Recommended)**

If you already have files split into 6-8 second chunks:

```
dataset.zip
├── train/
│   ├── audio001.wav
│   ├── audio002.wav
│   └── ... (235-250 files)
└── valid/
    ├── audio251.wav
    ├── audio252.wav
    └── ... (10-15 files)
```

**Option 2: Auto-split (Easy)**

Upload all your audio files in one folder - Colab will split and distribute them:

```
dataset.zip
└── (all your wav files here)
```

The notebook will automatically:
- Split long files into shorter segments
- Randomly assign 5-10 files to validation
- Extract features for training

---

## 🚀 Step 2: Open the Colab Notebook

1. **Click here:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fishaudio/fish-diffusion/blob/main/notebooks/train.ipynb)

2. **Select GPU Runtime:**
   - Menu: `Runtime` → `Change runtime type`
   - Hardware accelerator: **GPU** (T4)
   - Click `Save`

3. **Verify GPU is active:**
   - Run **Cell 5** (Check GPU)
   - You should see: `GPU 0: Tesla T4 detected`

---

## ⚙️ Step 3: Upload Dataset to Google Drive

### Upload Your ZIP

1. Open [Google Drive](https://drive.google.com)
2. Create folder: `Fish Diffusion`
3. Upload your `dataset.zip` to this folder
4. Verify path: `/MyDrive/Fish Diffusion/dataset.zip`

**Pro Tip:** Upload once, reuse forever! You can restart training anytime without re-uploading.

---

## 🔧 Step 4: Configure Training Settings

### Cell 18: Preprocess Dataset

```python
download_from_gdrive = True  # ✅ Keep as True
dataset_path = "Fish Diffusion/dataset.zip"  # ✅ Your path in Drive

auto_slice = False  # Set to True if using Option 2 above
auto_pick = True   # ✅ Automatically pick validation files
auto_pick_num = 15  # Number of validation files (10-15 recommended)
```

**Important Settings:**
- `auto_slice = False`: Use if you pre-split your files
- `auto_slice = True`: Let Colab split long files for you
- `auto_pick = True`: Randomly select validation files
- `auto_pick_num = 15`: Keep 15 files for validation (~6% of 250 files)

### Cell 20: Model Config

```python
pretrained = True  # ✅ Always use pretrained for best results!
pretrained_profile = 'hifisinger-v2.1.0'  # ✅ Recommended model
```

**Why Pretrained?**
- Starts from a model trained on thousands of hours
- Reaches excellent quality in 8k-16k steps (vs 100k+ from scratch)
- Your 30 min dataset fine-tunes the pretrained model
- **Config now caps training at 20,000 steps** (optimal range based on analysis)

---

## 🛠️ Step 5: Fix Precision Issue (CRITICAL!)

**After Cell 20**, add a **new code cell** with this fix:

```python
#@title Fix Precision for T4 GPU
# CRITICAL: T4 GPU doesn't support bf16 for STFT operations
# This patches the base config to use fp16 instead

import re

base_config = "/content/fish-diffusion/configs/_base_/trainers/base.py"
print(f"📝 Patching {base_config}...")

with open(base_config, 'r') as f:
    content = f.read()

# Replace bf16-mixed with 16-mixed (fp16)
original_content = content
content = re.sub(r'precision\s*=\s*["\']bf16-mixed["\']', 'precision="16-mixed"', content)

with open(base_config, 'w') as f:
    f.write(content)

if content != original_content:
    print("✅ Fixed! Changed bf16-mixed → 16-mixed (fp16)")
    print("   Base trainer config patched successfully")
else:
    print("ℹ️  No changes needed - precision already correct")

# Verify the change
with open(base_config, 'r') as f:
    if '16-mixed' in f.read():
        print("✅ Verified: Precision is now set to fp16")
```

**Why This is Needed:**
T4 GPUs don't support bfloat16 for certain audio operations. This fix prevents the error:
```
RuntimeError: cuFFT doesn't support tensor of type: BFloat16
```

---

## 🔧 Step 5b: Fix Checkpoint Management (IMPORTANT!)

**After the precision fix**, add another **new code cell** to fix checkpoint retention:

```python
#@title Fix Checkpoint Management - Keep Best 10 by Validation Loss
# This patches colab_train.py to keep the 10 BEST checkpoints by validation loss
# instead of only keeping the last 4 by time

import re

colab_train_file = "/content/fish-diffusion/tools/hifisinger/colab_train.py"
print(f"📝 Patching {colab_train_file}...")

with open(colab_train_file, 'r') as f:
    content = f.read()

# Add get_valid_loss function after get_step
if 'def get_valid_loss(' not in content:
    get_valid_loss_func = '''

def get_valid_loss(filename):
    """Extract validation loss from checkpoint filename."""
    match = re.search(r"valid_loss=([\\d.]+)", filename)
    if match:
        loss_str = match.group(1).rstrip('.')
        try:
            return float(loss_str)
        except ValueError:
            return float('inf')
    return float('inf')  # If no valid_loss in filename, treat as worst
'''

    content = content.replace(
        'def get_step(filename):\n    match = re.search(r"step=(\\d+)", filename)\n    return int(match.group(1)) if match else -1',
        'def get_step(filename):\n    match = re.search(r"step=(\\d+)", filename)\n    return int(match.group(1)) if match else -1' + get_valid_loss_func
    )
    print("  ✅ Added get_valid_loss() function")

# Fix local checkpoint cleanup (keep best 10 by loss)
old_local_cleanup = re.search(
    r'# Keep only the last four checkpoints.*?os\.remove\(checkpoint\)',
    content,
    re.DOTALL
)

if old_local_cleanup and 'best 10' not in old_local_cleanup.group(0):
    new_local_cleanup = '''# Keep only the best 10 checkpoints by validation loss in the source checkpoint directory
        for d in os.listdir(fishsvc_chkpt_path):
            checkpoints_path = os.path.join(fishsvc_chkpt_path, d, "checkpoints")
            if os.path.exists(checkpoints_path):
                checkpoints = [
                    os.path.join(checkpoints_path, f)
                    for f in os.listdir(checkpoints_path)
                    if f.endswith('.ckpt') and ".ipynb_checkpoints" not in f
                ]
                checkpoints_sorted = sorted(checkpoints, key=get_valid_loss)
                for checkpoint in checkpoints_sorted[10:]:
                    try:
                        os.remove(checkpoint)
                        logging.info("Deleted checkpoint (not in top 10): %s", os.path.basename(checkpoint))
                    except Exception as e:
                        logging.warning("Failed to delete %s: %s", checkpoint, e)'''

    content = re.sub(
        r'# Keep only the last four checkpoints.*?os\.remove\(checkpoint\)',
        new_local_cleanup,
        content,
        count=1,
        flags=re.DOTALL
    )
    print("  ✅ Fixed local checkpoint cleanup (keep best 10)")

# Fix Google Drive checkpoint cleanup (keep best 10 by loss)
old_drive_cleanup = re.search(
    r'# Keep only the last four checkpoints in the destination.*?os\.remove\(checkpoint\)',
    content,
    re.DOTALL
)

if old_drive_cleanup and 'best 10' not in old_drive_cleanup.group(0):
    new_drive_cleanup = '''# Keep only the best 10 checkpoints by validation loss in Google Drive
            models_path = os.path.join(fishsvc_dest_path, "models")
            if os.path.exists(models_path):
                checkpoints = [
                    os.path.join(models_path, f)
                    for f in os.listdir(models_path)
                    if f.endswith('.ckpt')
                ]
                checkpoints_sorted = sorted(checkpoints, key=get_valid_loss)
                for checkpoint in checkpoints_sorted[10:]:
                    try:
                        os.remove(checkpoint)
                        logging.info("Deleted Drive checkpoint (not in top 10): %s", os.path.basename(checkpoint))
                    except Exception as e:
                        logging.warning("Failed to delete Drive checkpoint %s: %s", checkpoint, e)'''

    content = re.sub(
        r'# Keep only the last four checkpoints in the destination.*?os\.remove\(checkpoint\)',
        new_drive_cleanup,
        content,
        count=1,
        flags=re.DOTALL
    )
    print("  ✅ Fixed Google Drive checkpoint cleanup (keep best 10)")

with open(colab_train_file, 'w') as f:
    f.write(content)

print("\n✅ Checkpoint management fixed!")
print("   • Keeps best 10 checkpoints by validation loss")
print("   • Captures both optimal peaks (7k-8k and 15k-16k)")
print("   • No manual downloads needed!")
```

**Why This is Needed:**
The default Colab training script only keeps the last 4 checkpoints by time, deleting earlier optimal checkpoints. This fix ensures you keep the 10 best checkpoints by validation loss, capturing both optimal training peaks automatically.

---

## 🎓 Step 6: Extract Features

**Run Cell 23** (Extract Features)

This processes your audio files:
- Extracts HuBERT soft features (text content)
- Extracts pitch using ParselMouth
- Creates `.npy` files for training

**Expected Output:**
```
Processing train: 100%|██████████| 252/252 [15:20<00:00]
Processing valid: 100%|██████████| 15/15 [00:55<00:00]
✅ Feature extraction complete!
```

**Time Required:** ~15-20 minutes for 250 files

---

## 🚂 Step 7: Start Training

**Run Cell 25** (Training)

### Key Parameters in Cell 25:

```python
logger = 'tensorboard'  # ✅ Recommended for monitoring
dest_path = '/content/drive/MyDrive/FishSVC/'  # ✅ Saves to Google Drive!
```

**What Happens:**
1. TensorBoard launches automatically
2. Training starts with pretrained HiFiSinger model
3. Checkpoints save to Google Drive every 1000 steps
4. Audio samples generated for validation

### Expected Output:
```
Using 16bit Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
102M Trainable params

Epoch 4:  91% 20/22 [00:09<00:00,  2.09it/s,
  train_loss_disc_step=3.17,
  train_loss_gen_step=110.0]
```

---

## 📊 Step 8: Monitor Training

### TensorBoard Tabs

**1. Audio Tab (MOST IMPORTANT!)**
- 🎧 Listen to "GT" (Ground Truth - original) vs "Prediction" (generated)
- **This is your real quality metric!**
- Check every 1000-2000 steps

**2. Scalars Tab**
- `train_loss_aux`: Should decrease steadily (stability indicator)
- `train_loss_gen_epoch`: Generator loss (decreases over time)
- `train_loss_disc_epoch`: Discriminator loss (stays ~3-5)
- `valid_loss_epoch`: Validation loss (will bounce - normal for GANs!)

**3. What to Expect:**

| Step | train_loss_gen | Audio Quality |
|------|----------------|---------------|
| 1,000 | ~100-110 | Recognizable but artifacts |
| 4,000 | ~50-80 | Very good, clear voice |
| 10,000 | ~20-40 | Excellent quality |
| 15,000 | ~10-30 | Peak quality |

---

## ⏱️ Step 9: Training Duration

### Automatic Training Cap

**Training now stops automatically at 20,000 steps** (configured in `svc_hifisinger_finetune.py`). This is optimal based on analysis:

- **Peak quality:** Steps 7,000-8,000 and 15,000-16,000
- **Overtraining:** Begins after ~8,000 steps (model may memorize artifacts)
- **20k cap:** Captures both optimal ranges with safety buffer

### When Quality is Best

✅ **Listen to audio samples regularly!** (most important)
- Clear voice reproduction
- Minimal artifacts
- Natural timbre

✅ **Validation loss is lowest**
- Check TensorBoard Scalars tab: `valid_loss_epoch`
- **Best range typically:** 0.942-0.948 (varies by dataset)

### Training Timeline

**T4 GPU (Free Colab):**
- ~2-3 seconds per batch
- ~8 batches per epoch (252 files, batch size 32)
- ~30 seconds per epoch
- **1,000 steps:** ~30-45 minutes
- **8,000 steps:** ~4-6 hours (first optimal peak)
- **16,000 steps:** ~8-10 hours (second optimal peak)
- **20,000 steps:** ~10-12 hours (automatic stop)

**Recommended Strategy:**
1. Let it run to completion (20k steps) - it will stop automatically
2. Monitor TensorBoard audio samples at steps 4k, 8k, 12k, 16k
3. After training, test multiple checkpoints (see Step 10)
4. Pick the best sounding one (usually step 7k-16k range)

---

## 💾 Step 10: Checkpoints & Backups

### Automatic Checkpoint Management (FIXED!)

**NEW: Keeps best 10 checkpoints by validation loss!**

The training script now automatically:
- Saves a checkpoint every 500 steps
- **Keeps the 10 BEST checkpoints** based on validation loss (lowest = best)
- Deletes checkpoints with higher validation loss
- Runs cleanup every 30 seconds

This means you'll always have the top 10 performing models saved automatically!

**⚠️ Important:** This fix requires patching `tools/hifisinger/colab_train.py`. See the fix cell in the notebook setup section.

### Where Are Checkpoints Saved?

**Google Drive Location:**
```
/MyDrive/FishSVC/
└── models/
    ├── epoch=X-step=7500-valid_loss=0.942.ckpt  ← Best 10 by loss
    ├── epoch=X-step=8000-valid_loss=0.942.ckpt  ← Often optimal
    ├── epoch=X-step=15500-valid_loss=0.940.ckpt ← Usually best
    ├── epoch=X-step=16000-valid_loss=0.940.ckpt
    ├── epoch=X-step=10000-valid_loss=0.945.ckpt
    └── ... (10 total checkpoints with lowest loss)
```

**Note:** The script keeps the 10 checkpoints with the **lowest validation loss**, regardless of when they were created. This ensures you keep checkpoints from both optimal peaks (7k-8k and 15k-16k)!

### Checkpoint Naming

```
epoch=178-step=7999-valid_loss=0.94.ckpt
  ^       ^           ^
  |       |           └─ Validation loss (lower = better)
  |       └─ Training step number
  └─ Epoch number
```

**Which checkpoint to use?**
- **The checkpoint with the LOWEST valid_loss** is typically the best
- Usually from step 7,500-8,000 (valid_loss ~0.942) or 15,500-16,000 (valid_loss ~0.940-0.942)
- All top 10 checkpoints are automatically preserved
- Listen to audio quality to confirm - validation loss correlates strongly with quality
- Compare your top 3-4 checkpoints to find the best sound

### After Training

All 10 best checkpoints are automatically saved to Google Drive. Simply:
1. Check `/content/drive/MyDrive/FishSVC/models/`
2. Sort by filename to find the checkpoint with lowest `valid_loss`
3. That's your best model!

---

## 🎤 Step 11: Test Your Model (Inference)

### After Training Completes

**Run Cell 28** (Gradio UI)

```python
checkpoint_path = "/content/fish-diffusion/logs/HiFiSVC/version_0/checkpoints"
# Will auto-find latest checkpoint
```

**What You Get:**
- Web interface for testing voice conversion
- Upload any audio → Convert to trained voice
- Adjust pitch, speaker mixing, etc.

### Using the Gradio Interface

1. **Upload Audio:** Your input voice (any speaker)
2. **Select Model:** Your trained model
3. **Adjust Pitch:** +/- 12 semitones
4. **Convert:** Click to generate output
5. **Download:** Save converted audio

---

## 🔧 Troubleshooting

### Training Errors

**Error: `RuntimeError: cuFFT doesn't support tensor of type: BFloat16`**
- **Solution:** You forgot Step 5 (precision fix)!
- Add the precision fix cell and restart training

**Error: `CUDA out of memory`**
- **Solution:** Reduce batch size in config
- Edit Cell 20 downloaded config: `batch_size=16` (from 32)

**Error: `No checkpoints folder found`**
- **Normal!** This warning appears until step 1000
- Ignore until you've trained for 1000 steps

### Dataset Issues

**Error: `Your dataset structure is incorrect`**
- Check your ZIP structure (see Step 1)
- Must have `train/` and `valid/` folders
- OR use `auto_slice=True` with no folders

**Error: `No validation files found`**
- Set `auto_pick=True` and `auto_pick_num=15`
- Or manually add files to `valid/` folder

### Quality Issues

**Audio sounds robotic/garbled:**
- Train longer (you're probably at <2000 steps)
- Check dataset quality (clean vocals?)
- Try different checkpoint (e.g., step 10k vs 5k)

**Audio has noise/artifacts:**
- Normal at early stages (<5000 steps)
- Check your source dataset quality
- Continue training to 10k-15k steps

---

## 📈 Expected Training Progress

### Real Results from 252 Files (30 min audio):

```
Step 1,000:  "Good quality, recognizable"
Step 4,000:  "Honestly hard to believe how good these sound!"
Step 8,000:  🏆 OPTIMAL - Best validation loss (0.9420)
Step 16,000: 🏆 OPTIMAL - Second peak, tied for best (0.9422)
Step 20,000: Training stops automatically
Step 54,000: ⚠️ Overtrained - reverb artifacts, less tight sound
```

### Loss Curves (Actual Data):

**valid_loss_epoch (MOST IMPORTANT):**
```
Step 1k:   0.9796  (improving)
Step 4k:   0.9475  ✅ (very good)
Step 8k:   0.9420  🏆 (BEST - absolute minimum)
Step 9k:   0.9505  ⚠️ (overtraining begins)
Step 16k:  0.9422  🏆 (recovery, nearly optimal)
Step 20k:  0.9475  (still decent, automatic stop)
Step 27k+: 0.948+  ⚠️ (degrading)
```

**Key Insight:** Validation loss reaches absolute minimum at step 7,999, then oscillates. The model briefly recovers around step 15,999 but never beats the step 7,999 checkpoint. **Always keep checkpoints from both peaks!**

---

## ⚡ Pro Tips

### Maximizing Free Colab Time

1. **Keep browser active:** Colab disconnects if idle too long
2. **Browser extension:** Use "Colab Auto-Clicker" to prevent timeout
3. **Monitor regularly:** Check every 2-3 hours
4. **Colab Pro ($10/mo):** 24+ hour sessions, faster GPUs

### Dataset Optimization

1. **Pre-split audio:** 6-8 second chunks = optimal training
2. **Validation selection:** Random is fine, 10-15 files sufficient
3. **Audio quality:** Clean > Quantity (30 min clean > 2hr noisy)

### Training Strategy

1. **Start small:** 5,000 steps first, listen to results
2. **Iterate:** Continue if needed, stop if excellent
3. **Compare checkpoints:** Listen to step 5k, 10k, 15k
4. **Trust your ears:** Audio quality > loss numbers!

---

## 🎯 Quick Reference

### Cell Execution Order

1. ✅ Cell 2: Agree to terms
2. ✅ Cell 5: Check GPU
3. ✅ Cells 8-14: Install dependencies (~5 minutes)
4. ✅ Cell 16: Download vocoder
5. ✅ Cell 18: Load & preprocess dataset (~20 minutes)
6. ✅ Cell 20: Select model (HiFiSinger pretrained)
7. ✅ **NEW CELL**: Precision fix (CRITICAL!)
8. ✅ Cell 23: Extract features (~15 minutes)
9. ✅ Cell 25: Start training! (8-12 hours to 15k steps)
10. ✅ Cell 28: Inference/testing

### Key Files & Locations

**Dataset in Drive:**
```
/MyDrive/Fish Diffusion/dataset.zip
```

**Checkpoints in Drive:**
```
/MyDrive/FishSVC/logs/HiFiSVC/version_X/checkpoints/
```

**Local workspace (temporary):**
```
/content/fish-diffusion/dataset/
/content/fish-diffusion/logs/
```

---

## 🎉 Success Checklist

- [ ] Dataset prepared and uploaded to Drive
- [ ] GPU runtime selected (T4)
- [ ] Precision fix applied (Step 5)
- [ ] Features extracted successfully
- [ ] Training running without errors
- [ ] TensorBoard showing progress
- [ ] Audio samples sound excellent at step 5k-10k
- [ ] Checkpoints saving to Google Drive
- [ ] Final model tested with Gradio UI

**Congratulations! You've trained a custom voice conversion model!** 🎤🎵

---

## 📞 Getting Help

**Issues or Questions?**

- 💬 [Discord Community](https://discord.gg/wbYSRBrW2E)
- 🐛 [GitHub Issues](https://github.com/fishaudio/fish-diffusion/issues)
- 📚 [Official Wiki](https://fishaudio.github.io/fish-diffusion/)

**Share Your Results!**
We'd love to hear about your training success - share on Discord!

---

*Last updated: October 2025*
