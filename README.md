# MARGOT - Optimized Fish-Diffusion for Voice Conversion

<div align="center">
  <img src="https://cdn.jsdelivr.net/gh/fishaudio/fish-diffusion@main/images/logo_512x512.png" width="256" height="256" alt="Fish Diffusion Logo"/>
</div>

<div align="center">

[![Open In Colab](https://img.shields.io/static/v1?label=Colab&message=Train%20Now&color=F9AB00&logo=googlecolab&style=flat-square)](https://colab.research.google.com/github/gdoscher/MARGOT/blob/main/notebooks/train.ipynb)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?style=flat-square)](LICENSE)
[![Discord](https://img.shields.io/discord/1044927142900809739?color=%23738ADB&label=Fish%20Audio%20Discord&logo=discord&logoColor=white&style=flat-square)](https://discord.gg/wbYSRBrW2E)

**Production-ready voice conversion training optimized for Google Colab**

[Quick Start](#-quick-start) • [What's Different](#-whats-different-from-original) • [Training Guide](COLAB_TRAINING_GUIDE.md) • [Results](#-expected-results)

</div>

---

## 🎯 What is MARGOT?

**MARGOT** is an optimized fork of [fish-diffusion](https://github.com/fishaudio/fish-diffusion) specifically configured for hassle-free voice conversion training on **Google Colab's free T4 GPU**.

Train a custom voice conversion model with just **30 minutes of audio** in **8-12 hours** - no local GPU required!

### Key Improvements

✅ **Works out-of-the-box on Colab T4 GPUs** - fp16 precision (no more bf16 errors)
✅ **Smart checkpoint management** - keeps best 10 models by validation loss (4-decimal precision)
✅ **Intelligent training duration** - auto-recommends steps based on your dataset size
✅ **Optimized validation frequency** - 500-step intervals capture both quality peaks (7k-8k & 15k-16k)
✅ **Comprehensive training guide** - step-by-step instructions with troubleshooting
✅ **All fixes permanent** - no runtime patching required

---

## ⚡ Quick Start

### 1️⃣ Prepare Your Dataset

Create a ZIP file with your audio:

```
dataset.zip
├── train/
│   └── *.wav files (6-8 sec each, 30-60 min total)
└── valid/
    └── *.wav files (5-10 samples)
```

**Requirements:**
- Single speaker only
- Clean vocals (no background music)
- WAV format, 44.1kHz recommended
- 30+ minutes total (more is better!)

### 2️⃣ Upload to Google Drive

1. Go to [Google Drive](https://drive.google.com)
2. Create folder: `Fish Diffusion`
3. Upload your `dataset.zip`
4. Path should be: `/MyDrive/Fish Diffusion/dataset.zip`

### 3️⃣ Start Training on Colab (FREE!)

**Click here:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gdoscher/MARGOT/blob/main/notebooks/train.ipynb)

**Select GPU Runtime:**
- Runtime → Change runtime type → GPU (T4) → Save

**Run the cells in order - that's it!** 🎉

All optimizations are built-in. No manual fixes needed.

### 4️⃣ Monitor & Download

- **TensorBoard** launches automatically - listen to audio samples!
- Training stops automatically at optimal step count (typically 20k steps)
- **Best 10 checkpoints** saved to Google Drive at `/MyDrive/FishSVC/models/`
- Use the checkpoint with the **lowest validation loss** (e.g., `valid_loss=0.9400`)

📖 **Full guide:** [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md)

---

## 📊 Expected Results

### Real Results from 252 Files (~30 min audio):

| Steps | Quality | Valid Loss | Notes |
|-------|---------|------------|-------|
| 1,000 | Good | 0.9796 | Recognizable voice |
| 4,000 | Very Good | 0.9475 | *"Hard to believe how good!"* |
| **7,999** | **🏆 PEAK #1** | **0.9420** | **Absolute best - first optimal peak** |
| 9,000 | Excellent | 0.9505 | Slight overtraining begins |
| **15,999** | **🏆 PEAK #2** | **0.9422** | **Nearly optimal - second peak** |
| 20,000 | Very Good | 0.9475 | Auto-stops, still excellent |
| 54,000+ | ⚠️ Overtrained | 0.948+ | Reverb artifacts, less tight |

**Key Insight:** Validation loss hits absolute minimum at step 7,999, then oscillates. Model recovers around step 15,999 but never quite beats the first peak. **MARGOT keeps both automatically!**

### Timeline (T4 GPU)

- **1,000 steps:** ~45 minutes
- **8,000 steps:** ~4-6 hours (first optimal peak)
- **16,000 steps:** ~8-10 hours (second optimal peak)
- **20,000 steps:** ~10-12 hours (automatic stop)

---

## 🆚 What's Different from Original?

This fork includes production-ready optimizations for Colab training:

### Configuration Changes

| Setting | Original | MARGOT | Why? |
|---------|----------|--------|------|
| **Precision** | bf16-mixed | **fp16 (16-mixed)** | T4 GPU compatibility |
| **Max Steps** | 2,000,000 | **20,000** | Optimal for pretrained fine-tuning |
| **Validation Frequency** | 5,000 steps | **500 steps** | Captures quality peaks precisely |
| **Checkpoint Frequency** | Varies | **500 steps** | 2x granularity |
| **Checkpoint Retention** | Last 4 by time | **Best 10 by loss** | Keeps optimal models |
| **Loss Precision** | 2 decimals (.2f) | **4 decimals (.4f)** | Proper sorting (0.9420 vs 0.9422) |

### New Features

✨ **Smart Training Steps Cell** - Analyzes dataset size and recommends optimal duration:
- < 150 files (~15 min): 10,000 steps
- 150-300 files (~30 min): 20,000 steps
- 300-450 files (~45 min): 25,000 steps
- > 450 files (~60+ min): 30,000 steps

✨ **Comprehensive Training Guide** - [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md) with:
- Complete step-by-step instructions
- Troubleshooting section
- Expected results and timelines
- HiFiSinger vs DiffSVC comparison

✨ **Updated Colab Notebook** - Clones from MARGOT repo with all fixes built-in

### Files Modified

- `configs/_base_/trainers/base.py` - Core training settings
- `configs/svc_hifisinger_finetune.py` - HiFiSinger-specific config
- `tools/hifisinger/colab_train.py` - Checkpoint management
- `notebooks/train.ipynb` - Training notebook
- `COLAB_TRAINING_GUIDE.md` - Comprehensive guide (new)
- `PROJECT_SUMMARY.md` - Technical analysis (new)

---

## 🎓 Model Selection: HiFiSinger vs DiffSVC

**TL;DR: Use HiFiSinger v2.1.0** (the default)

### HiFiSinger v2.1.0 (RECOMMENDED)

- **Type:** GAN-based (fast)
- **Best for:** 30-60 min datasets
- **Training time:** 8-12 hours to peak quality
- **Inference:** Real-time capable
- **Quality:** Excellent at 8k-16k steps

### DiffSVC v2.0.0 (Legacy)

- **Type:** Diffusion-based (slow but stable)
- **Best for:** 100+ min datasets, research
- **Training time:** 20+ hours (50k-100k steps)
- **Inference:** Slower (iterative denoising)
- **Quality:** Highest possible, but requires much more time

**For Colab free tier with limited datasets, HiFiSinger is the clear choice.**

---

## 📚 Documentation

- **[COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md)** - Complete training walkthrough
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Technical analysis and decisions
- **[Original Wiki](https://fishaudio.github.io/fish-diffusion/)** - Fish-Diffusion documentation

---

## 🛠️ Advanced: Local Training

<details>
<summary>Click to expand local installation instructions</summary>

### Prerequisites

- Python 3.10
- CUDA-compatible GPU (8GB+ VRAM)
- Linux or macOS

### Installation

```bash
# Clone MARGOT repository
git clone https://github.com/gdoscher/MARGOT.git
cd MARGOT

# Install dependencies using PDM
pip install pdm
pdm install

# Download pretrained vocoder
python tools/download_nsf_hifigan.py --agree-license
```

### Prepare Dataset

```bash
# Extract features
python tools/preprocessing/extract_features.py \
    --config configs/svc_hifisinger_finetune.py \
    --path dataset/train \
    --clean \
    --num-workers 4
```

### Train Locally

```bash
python tools/hifisinger/train.py \
    --config configs/svc_hifisinger_finetune.py \
    --pretrain checkpoints/hifisinger-pretrained.ckpt
```

Monitor with TensorBoard:
```bash
tensorboard --logdir logs/
```

</details>

---

## 🤝 Credits & License

### MARGOT Fork

- **Maintainer:** [@gdoscher](https://github.com/gdoscher)
- **Purpose:** Production-ready Colab training optimizations
- **Based on:** [fish-diffusion](https://github.com/fishaudio/fish-diffusion) by Fish Audio

### Original Fish-Diffusion

- **Authors:** Fish Audio Team
- **Repository:** https://github.com/fishaudio/fish-diffusion
- **Discord:** https://discord.gg/wbYSRBrW2E

### License

This project inherits the [CC BY-NC-SA 4.0](LICENSE) license from fish-diffusion:

- ✅ You can use for research and personal projects
- ✅ You must credit Fish Audio and this fork
- ✅ Share-alike: derivatives must use same license
- ❌ No commercial use without permission

**Attribution Required:** When sharing models or results, credit both:
- Original project: Fish-Diffusion by Fish Audio
- This fork: MARGOT optimized configurations

---

## 📞 Support & Contributing

### Getting Help

- 📖 Read [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md) first
- 💬 Join [Fish Audio Discord](https://discord.gg/wbYSRBrW2E) for community support
- 🐛 [Open an issue](https://github.com/gdoscher/MARGOT/issues) for bugs

### Contributing

Contributions welcome! Areas of interest:

- Further Colab optimizations
- Dataset preprocessing improvements
- Training efficiency enhancements
- Documentation improvements

---

## ⚠️ Terms of Use

1. **Authorization Required:** You are responsible for obtaining permission to use any voice data in your training dataset
2. **Attribution:** Credit Fish-Diffusion and declare AI-generated content when sharing results
3. **Ethical Use:** Do not create deepfakes or use for malicious purposes
4. **No Liability:** Neither Fish-Diffusion nor MARGOT developers are responsible for misuse

By using MARGOT, you agree to use it responsibly and ethically.

---

<div align="center">

**Ready to train your voice model?**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gdoscher/MARGOT/blob/main/notebooks/train.ipynb)

⭐ **Star this repo if MARGOT helped you!** ⭐

</div>
