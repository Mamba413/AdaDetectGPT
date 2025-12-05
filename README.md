# AdaDetectGPT

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue)](https://neurips.cc/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

This repository contains the implementation of [**AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees**](https://arxiv.org/abs/2510.01268), presented at NeurIPS 2025. Our method provides adaptive detection of LLM-generated text with statistical guarantees. We build upon and extend code from [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt).

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

A quick example is: 
```bash
python scripts/local_infer_ada.py \
  --text "Your text to be detected" \
  --train_dataset "./exp_gpt3to4/data/essay_claude-3-5-haiku&./exp_gpt3to4/data/xsum_claude-3-5-haiku"
```

### Without Training Data

AdaDetectGPT can also use pretrained parameters (trained on texts from GPT-4o, Gemini-2.5, and Claude-3.5):

```bash
python scripts/local_infer_ada.py --text "Your text to be detected"
```

## 🔬 Reproducibility

We provide generated text samples from GPT-3.5-Turbo, GPT-4, GPT-4o, Gemini-2.5, and Claude-3.5 in `exp_gpt3to4/data/` for convenient reproduction. Data from GPT-3.5-Turbo and GPT-4 are sourced from [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt).

### Experiment Scripts

#### White-box Experiments
- `./exp_whitebox.sh` - Table 1: Evaluation on 5 base LLMs
  - GPT-2 (1.5B), GPT-Neo (2.7B), OPT-2.7B, GPT-J (6B), GPT-NeoX (20B)
- `./exp_whitebox_advanced.sh` - Table S7: Advanced open-source LLMs
  - Qwen-2.5 (7B), Mistral (7B), Llama3 (8B)

#### Black-box Experiments
- `./exp_blackbox_advanced.sh` - Table 2 and Table S8: Advanced closed-source LLMs
  - Gemini-2.5-Flash, GPT-4o, Claude-3.5-Haiku
- `./exp_blackbox_simple.sh` - Table S2: Five open-source LLMs

#### Analysis Experiments
- `./exp_attack.sh` - Table 3: Adversarial attack evaluation
- `./exp_normal.sh` - Data for Figure 3 and Figure S8
- `./exp_sample.sh` - Training data size effects (Figure S5)
- `./exp_tuning.sh` - Hyperparameter robustness (Figure S6)
- `./exp_dist_shift.sh` - Distribution shift analysis (Figure S7)
- `./exp_compute.sh` - Computational cost analysis (Table S9 and S10)
- `./exp_variance.sh` - Equal variance condition verification (Table S5)

## 🎁 Additional Resources

The `scripts/` directory contains implementations of various LLM detection methods from the literature. These implementations are modified from their official versions or the repo of [FastDetectGPT](https://github.com/baoguangsheng/fast-detect-gpt) to provide:
- Consistent input/output formats
- Simplified method comparison

The provided methods are summarized below.

| Method | Script File | Paper/Website |
|--------|------------|---------------|
| **AdaDetectGPT** | `detect_gpt_ada.py` | [arXiv:2510.01268](https://arxiv.org/abs/2510.01268) |
| **Binoculars** | `detect_binoculars.py` | [arXiv:2401.12070](https://arxiv.org/abs/2401.12070) |
| **BiScope** | `detect_biscope.py` | [NeurIPS 2024](https://neurips.cc/virtual/2024/poster/95814) |
| **DetectGPT** | `detect_gpt.py` | [arXiv:2301.11305](https://arxiv.org/abs/2301.11305) |
| **DetectLLM** | `detect_llm.py` | [arXiv:2306.05540](https://arxiv.org/abs/2306.05540) |
| **DNA-GPT** | `detect_gpt_dna.py` | [arXiv:2305.17359](https://arxiv.org/abs/2305.17359) |
| **Fast-DetectGPT** | `detect_gpt_fast.py` | [arXiv:2310.05130](https://arxiv.org/abs/2310.05130) |
| **GLTR** | `detect_gltr.py` | [arXiv:1906.04043](https://arxiv.org/abs/1906.04043) |
| **ImBD** | `detect_ImBD.py` | [arXiv:2412.10432](https://arxiv.org/abs/2412.10432) |
| **GPTZero** | `detect_gptzero.py` | [GPTZero.me](https://gptzero.me/) |
| **RADAR** | `detect_radar.py` | [arXiv:2307.03838](https://arxiv.org/abs/2307.03838) |
| **RoBERTa OpenAI Detector** | `detect_roberta.py` | [arXiv:1908.09203](https://arxiv.org/abs/1908.09203) |
| **Text Fluoroscopy** | `detect_fluoroscopy.py` | [EMNLP 2024](https://aclanthology.org/2024.emnlp-main.885/) |

We hope these resources facilitate your research and applications in LLM-generated text detection!

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

For any questions/suggestions/bugs, feel free to open an [issue](https://github.com/Mamba413/AdaDetectGPT/issues) in the repository.
