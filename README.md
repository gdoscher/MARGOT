# Fish-Diffusion: Voice Conversion Training

<div align="center">
  <img src="https://cdn.jsdelivr.net/gh/fishaudio/fish-diffusion@main/images/logo_512x512.png" width="256" height="256" alt="Fish Diffusion Logo"/>
</div>

<div align="center">

[![Discord](https://img.shields.io/discord/1044927142900809739?color=%23738ADB&label=Discord&logo=discord&logoColor=white&style=flat-square)](https://discord.gg/wbYSRBrW2E)
[![Open In Colab](https://img.shields.io/static/v1?label=Colab&message=Train%20Now&color=F9AB00&logo=googlecolab&style=flat-square)](https://colab.research.google.com/github/fishaudio/fish-diffusion/blob/main/notebooks/train.ipynb)
[![Hugging Face](https://img.shields.io/badge/🤗%20Spaces-HiFiSinger-blue.svg?style=flat-square)](https://huggingface.co/spaces/fishaudio/fish-diffusion)

</div>

---

## 🎯 Overview

Fish-Diffusion is a state-of-the-art voice conversion system using **HiFiSinger** (GAN-based) and **DiffSVC** (diffusion-based) architectures. This repository contains tools for training custom voice conversion models on your own datasets.

**Recommended Approach:** Use the official **Google Colab notebook** for GPU training - no local setup required!

---

## ⚡ Quick Start (Recommended)

### 1. Prepare Your Dataset Locally
```bash
# Your audio files should be:
# - Single speaker
# - Clean vocals (no background music)
# - WAV format (44.1kHz recommended)
# - 30+ minutes total duration

# Upload as ZIP with this structure:
dataset.zip
├── train/
│   └── *.wav files
└── valid/
    └── *.wav files (5-10 files recommended)
```

### 2. Train on Google Colab (FREE GPU!)

**Click here:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fishaudio/fish-diffusion/blob/main/notebooks/train.ipynb)

**Key Settings:**
- **Model:** HiFiSinger v2.1.0 (pretrained, recommended)
- **Dataset:** Upload your `dataset.zip` to Google Drive at `/MyDrive/Fish Diffusion/dataset.zip`
- **Training Duration:** Automatic stop at 20,000 steps (~10-12 hours on T4 GPU)
- **Checkpoints:** Best 10 checkpoints saved automatically (by validation loss)
- **Important:** Apply the checkpoint management fix from the training guide (Step 5b)

### 3. Monitor Training

TensorBoard launches automatically in the notebook. Monitor:
- 🎧 **Audio Tab:** Listen to prediction samples (most important!)
- 📊 **train_loss_aux:** Should decrease steadily
- 📉 **valid_loss_epoch:** Lower = better quality (bounces for GAN models)

### 4. Training Completes Automatically

**Training stops automatically at 20,000 steps** (optimal range). The config keeps the **best 10 checkpoints** based on validation loss, so you'll always have the highest quality models saved!

---

## 📊 Expected Results

### With Optimized Dataset (30 min audio, 6-8 sec chunks):

| Steps | Quality | Valid Loss | Notes |
|-------|---------|------------|-------|
| 1,000 | Good | 0.980 | Recognizable voice, some artifacts |
| 4,000 | Very Good | 0.948 | Clear voice, minimal artifacts |
| **8,000** | **🏆 Optimal** | **0.942** | **Peak quality - absolute best** |
| 12,000 | Excellent | 0.952 | High quality, slight overtraining |
| **16,000** | **🏆 Optimal** | **0.942** | **Second peak - tied for best** |
| 20,000 | Very Good | 0.948 | Auto-stops, still good quality |

**Real User Results:**
- *"Honestly hard to believe how good these prediction files sound at step 4000, this is crazy."*
- Step 8,000 and 16,000 checkpoints typically produce the best results
- Overtraining starts after step 8,000 but model recovers around step 16,000

---

## 🛠️ Advanced: Local Setup

<details>
<summary>Click to expand local installation instructions</summary>

### Prerequisites
- Python 3.10
- CUDA-compatible GPU (8GB+ VRAM recommended)
- Linux or macOS

### Installation
```bash
# Clone repository
git clone https://github.com/fishaudio/fish-diffusion.git
cd fish-diffusion

# Install PDM
curl -sSL https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py | python3 -

# Install dependencies
pdm sync
```

### Vocoder Preparation
```bash
# Download NSF-HiFiGAN vocoder (required for DiffSVC)
python tools/download_nsf_hifigan.py --agree-license
```

### Local Training
```bash
# Extract features
python tools/preprocessing/extract_features.py \
    --config configs/svc_hifisinger_finetune.py \
    --path dataset/train --clean

python tools/preprocessing/extract_features.py \
    --config configs/svc_hifisinger_finetune.py \
    --path dataset/valid --clean --no-augmentation

# Train
python tools/hifisinger/train.py \
    --config configs/svc_hifisinger_finetune.py \
    --pretrain checkpoints/hifisinger-pretrained.ckpt \
    --tensorboard
```

### Local Inference
```bash
# Using Gradio UI
python tools/hifisinger/inference.py \
    --config configs/svc_hifisinger_finetune.py \
    --checkpoint logs/HiFiSVC/version_0/checkpoints/best.ckpt \
    --gradio --gradio_share

# Command line
python tools/hifisinger/inference.py \
    --config configs/svc_hifisinger_finetune.py \
    --checkpoint logs/HiFiSVC/version_0/checkpoints/best.ckpt \
    --input input.wav \
    --output output.wav
```

</details>

---

## 📚 Documentation

- **[Colab Training Guide](COLAB_TRAINING_GUIDE.md)** - Detailed Colab workflow with troubleshooting
- **[Project Summary](PROJECT_SUMMARY.md)** - Technical architecture and model details
- **[Official Wiki](https://fishaudio.github.io/fish-diffusion/)** - Complete documentation

---

## 🎵 Model Architectures

### HiFiSinger (Recommended)
- **Type:** GAN-based (Generator + Discriminator)
- **Speed:** Fast training and inference
- **Quality:** Excellent with pretrained model
- **Best For:** Quick results, fine-tuning on small datasets

### DiffSVC
- **Type:** Diffusion-based
- **Speed:** Slower training/inference
- **Quality:** Very high quality, more stable
- **Best For:** Maximum quality, larger datasets

---

## 💡 Tips for Best Results

### Dataset Quality
✅ **Do:**
- Use clean vocal recordings (no background music)
- Consistent recording quality
- 30+ minutes total audio
- Diverse phonetic content
- 6-8 second audio chunks

❌ **Don't:**
- Mix multiple recording qualities
- Include silence/noise
- Use compressed audio formats (MP3)
- Use clips longer than 15 seconds

### Training Duration
- **HiFiSinger with pretrained:** 10,000-15,000 steps typical
- **DiffSVC from scratch:** 50,000-100,000 steps
- **Monitor audio quality** - numbers can be misleading for GANs!

### Hardware
- **Minimum:** T4 GPU (15GB VRAM) - Free on Google Colab!
- **Recommended:** V100 or A100 for faster training
- **Batch Size:** 32 for HiFiSinger, 16 for DiffSVC

---

## 📄 Terms of Use

1. **Authorization Required:** You are solely responsible for obtaining rights to any training data used and assume full responsibility for any infringement issues.

2. **Proper Attribution:** All derivative works must explicitly acknowledge Fish-Diffusion and its license. You must cite the original author and source code.

3. **AI-generated Disclosure:** All derivative works must declare that content is AI-generated and acknowledge the Fish-Diffusion project.

4. **Agreement to Terms:** By using Fish-Diffusion, you unequivocally consent to these terms. Neither Fish-Diffusion nor its developers shall be held liable for any subsequent difficulties that may transpire.

---

## 🤝 Contributing

If you have any questions, please submit an issue or pull request.
You should run `pdm run lint` before submitting a pull request.

Real-time documentation can be generated by:
```bash
pdm run docs
```

---

## ⭐ Support

If you find this project useful, please:
- ⭐ Star this repository
- 💬 Join our [Discord community](https://discord.gg/wbYSRBrW2E)
- 🐛 Report issues on GitHub
- 📢 Share your results!

---

## 🙏 Credits

+ [diff-svc original](https://github.com/prophesier/diff-svc)
+ [diff-svc optimized](https://github.com/innnky/diff-svc/)
+ [DiffSinger](https://github.com/openvpi/DiffSinger/) [Paper](https://arxiv.org/abs/2105.02446)
+ [so-vits-svc](https://github.com/innnky/so-vits-svc)
+ [iSTFTNet](https://github.com/rishikksh20/iSTFTNet-pytorch) [Paper](https://arxiv.org/pdf/2203.02395.pdf)
+ [HiFi-GAN](https://github.com/jik876/hifi-gan) [Paper](https://arxiv.org/abs/2010.05646)
+ [Retrieval-based-Voice-Conversion](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

---

## Thanks to all contributors

<a href="https://github.com/fishaudio/fish-diffusion/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=fishaudio/fish-diffusion" />
</a>

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong>Made with ❤️ by Fish Audio</strong>
  <br/>
  <a href="https://github.com/fishaudio/fish-diffusion">GitHub</a> •
  <a href="https://discord.gg/wbYSRBrW2E">Discord</a> •
  <a href="https://fishaudio.github.io/fish-diffusion/">Docs</a>
</div>
