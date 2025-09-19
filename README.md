# AdaDetectGPT

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue)](https://neurips.cc/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

This repository contains the implementation of **Adaptive Detection of LLM-Generated Text with Statistical Guarantees**, presented at NeurIPS 2025. Our method provides adaptive detection of LLM-generated text with statistical guarantees. We build upon and extend code from [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt).

## 📋 Overview

![AdaDetectGPT Workflow](figure/AdaDetectGPT.png)

Workflow of **AdaDetectGPT**. Built upon Fast-DetectGPT (Bao et al., 2024), our method adaptively learn a witness function $\hat{w}$ from training data by maximizing a lower bound on the TNR, while using normal approximation for FNR control.

## 🛠️ Installation

### Requirements
- Python 3.10.8
- PyTorch 2.7.0
- CUDA-compatible GPU (experiments conducted on H20-NVLink with 96GB memory)

### Setup
```bash
bash setup.sh
```

*Note: While our experiments used high-memory GPUs, typical usage of AdaDetectGPT requires significantly less memory.*

## 💻 Usage

### With Training Data (Recommended)

For optimal performance, we recommend using training data. The training dataset should be a `.json` file named `xxx.raw_data.json` with the following structure:

```json
{
  "original": ["human-text-1", "human-text-2", "..."],
  "sampled": ["machine-text-1", "machine-text-2", "..."]
}
```

Run detection with training data:
```bash
python scripts/local_infer_ada.py \
  --text "Your text to be detected" \
  --train_dataset "train-data-file-name"  ## for multiple training datasets, separate them with `&`
```

A quick example with 
```bash
python scripts/local_infer_ada.py \
  --text "Your text to be detected" \
  --train_dataset "dataset1&dataset2"
```

### Without Training Data

AdaDetectGPT can also operate using pretrained parameters (trained on outputs from GPT-4o, Gemini-2.5, and Claude-3.5):

```bash
python scripts/local_infer_ada.py --text "Your text to be detected"
```

## 🔬 Reproducibility

We provide generated text samples from GPT-3.5-Turbo, GPT-4, GPT-4o, Gemini-2.5, and Claude-3.5 in `exp_gpt3to4/data/` for convenient reproduction. Data from GPT-3.5-Turbo and GPT-4 are sourced from [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt).

### Experiment Scripts

#### White-box Experiments
- `./exp_whitebox.sh` - Table 1: Evaluation on 5 base LLMs
  - GPT-2 (1.5B), GPT-Neo (2.7B), OPT-2.7B, GPT-J (6B), GPT-NeoX (20B)
- `./exp_whitebox_advanced.sh` - Advanced open-source LLMs
  - Qwen-2.5 (7B), Mistral (7B), Llama3 (8B)

#### Black-box Experiments
- `./exp_blackbox.sh` - Table 2: GPT-4 and GPT-3.5-Turbo evaluation
- `./exp_blackbox_advanced.sh` - Advanced closed-source LLMs
  - Gemini-2.5-Flash, GPT-4o, Claude-3.5-Haiku
- `./exp_blackbox_simple.sh` - Table S2: Five open-source LLMs

#### Analysis Experiments
- `./exp_sample.sh` - Training data size effects (Figure S2)
- `./exp_tuning.sh` - Hyperparameter robustness (Figure S3)
- `./exp_dist_shift.sh` - Distribution shift analysis (Figure S4)
- `./exp_attack.sh` - Adversarial attack evaluation
- `./exp_compute.sh` - Computational cost analysis
- `./exp_variance.sh` - Equal variance condition verification
- `./exp_normal.sh` - Data for Figure 3 and Figure S1

## 🎁 Additional Resources

The `scripts/` directory contains unified implementations of various LLM detection methods from the literature. These implementations provide:
- Consistent input/output formats
- Simplified method comparison
- Easy integration for future research

We hope these resources facilitate your research in LLM-generated text detection!

## 📖 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{zhou2025adadetect,
  title={AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees},
  author={Hongyi Zhou and Jin Zhu and Pingfan Su and Kai Ye and Ying Yang and Shakeel A O B Gavioli-Akilagun and Chengchun Shi},
  booktitle={The Thirty-Ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```

## 📧 Contact

For questions or feedback, please open an issue in this repository.