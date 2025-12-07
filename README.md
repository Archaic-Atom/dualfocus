# DualFocus: A Framework for Efficient Decoupling of Visual Representation and Focus Fusion in Video LLM

<!-- [![Paper](https://img.shields.io/badge/Paper-PDF-red)](Your_Paper_Link_Here) -->
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Mingyu Sun, Hongwei Zhou, Zhiqing Luo, Xing Li, Jinhao Zhu, Tianwei Yan, and Zhibo Rao**

*Nanchang Hangkong University | Chongqing University | Beihang University | Chongqing Jiaotong University*

---

## 📢 Introduction

This repository contains the official implementation of the paper **"DualFocus: A Framework for Efficient Decoupling of Visual Representation and Focus Fusion in Video LLM"**.

**DualFocus** is a novel Video Large Language Model framework designed to address the challenges of information coupling and feature redundancy in existing approaches. Instead of relying on a single visual stream, DualFocus decouples video information into two distinct but interrelated streams:
1.  **Semantic Stream:** Extracts fine-grained, text-guided visual details using QA-ViT and Q-Former.
2.  **Spatio-temporal Stream:** Captures global dynamics and motion using a compressed ViT pathway.

To synergize these representations, we introduce the **Focus Feature Projector (FFP)**, which integrates the dual-stream information through an interactive attention mechanism. This results in a high-quality visual representation using only **32 tokens per second**, achieving State-of-the-Art (SOTA) performance on major video benchmarks while significantly reducing computational overhead.

<div align="center">
    <img src="assets/framework.svg" alt="DualFocus Framework" width="100%">
    <br>
    <em>Figure 1: The overall architecture of DualFocus. The video is processed in parallel by Semantic and Spatio-temporal streams, then fused by the Focus Feature Projector (FFP) before entering the LLM.</em>
</div>


## 🏆 Performance

We evaluate DualFocus on comprehensive benchmarks, including four open-ended Zero-shot VideoQA datasets and the Video-ChatGPT generation benchmark.

### 1. Zero-shot Video Question Answering
Comparison with SOTA methods on **MSVD-QA**, **MSRVTT-QA**, **ActivityNet-QA**, and **TGIF-QA**. We report both Accuracy (%) and Score (0-5).
*   **DualFocus** achieves SOTA performance across all datasets **without** fine-tuning the LLM (keeping Vicuna-7B frozen).

| Method | MSVD-QA<br>(Acc / Score) | MSRVTT-QA<br>(Acc / Score) | ActivityNet-QA<br>(Acc / Score) | TGIF-QA<br>(Acc / Score) |
| :--- | :---: | :---: | :---: | :---: |
| Video-ChatGPT | 64.9 / 3.3 | 49.3 / 2.8 | 35.2 / 2.7 | 40.7 / 3.1 |
| SlowFocus | 70.1 / 3.9 | 58.3 / 3.5 | 48.4 / 3.6 | - |
| VISTA-LLAMA | 65.3 / 3.6 | 60.5 / 3.3 | 48.3 / 3.3 | - |
| MiniGPT4-V | 72.9 / 3.8 | 58.8 / 3.3 | 45.9 / 3.2 | 67.9 / 3.7 |
| Chat-UniVi | 65.0 / 3.6 | 54.6 / 3.1 | 45.8 / 3.2 | 60.3 / 3.4 |
| VISTA | 71.5 / 4.0 | 58.5 / 3.5 | 49.1 / 3.4 | 71.4 / 4.0 |
| FarSight | 73.8 / 3.9 | - | 50.4 / **3.6** | - |
| Video-Panda | 64.7 / 3.8 | 54.8 / 3.4 | 40.0 / 3.3 | 42.9 / 3.2 |
| GASET | 73.4 / 3.9 | 59.7 / 3.3 | 51.4 / 3.4 | 74.9 / 4.1 |
| **DualFocus (Ours)** | **74.8** / **4.1** | **62.7** / **3.6** | **51.9** / 3.5 | **77.6** / **4.2** |

### 2. Video-Based Text Generation
Performance on the **Video-ChatGPT Benchmark** across five key dimensions:
*   **CI**: Correctness of Information
*   **DO**: Detail Orientation
*   **CU**: Contextual Understanding
*   **TU**: Temporal Understanding
*   **CO**: Consistency

| Method | CI | DO | CU | TU | CO | **Avg.** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Video-ChatGPT | 2.40 | 2.52 | 2.62 | 1.98 | 2.37 | 2.39 |
| SlowFocus | 2.95 | 3.03 | 3.61 | 2.54 | 2.60 | 2.95 |
| VISTA-LLAMA | 2.44 | 2.64 | 3.18 | 2.26 | 2.31 | 2.57 |
| LongVLM | 2.76 | 2.86 | 3.34 | 2.39 | 3.11 | 2.89 |
| MiniGPT4-V | 2.93 | 2.97 | 3.45 | 2.47 | 2.60 | 2.88 |
| Chat-UniVi | 2.89 | 2.91 | 3.46 | 2.40 | 2.81 | 2.89 |
| FarSight | 2.86 | 2.56 | 3.19 | 2.48 | 2.94 | 2.81 |
| Video-Panda | 2.74 | 2.47 | 3.01 | 2.26 | 2.36 | 2.57 |
| GASET | 3.07 | 3.04 | 3.62 | 2.56 | 2.78 | 3.01 |
| **DualFocus (Ours)** | **3.29** | **3.11** | **3.69** | **2.66** | **3.30** | **3.21** |

---
## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Archaic-Atom/dualfocus.git
    cd dualfocus
    ```

2.  **Create a Conda environment:**
    ```bash
    conda create -n dualfocus python=3.10
    conda activate dualfocus
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Ensure you have `accelerate` and `deepspeed` correctly configured for your system.

## 📦 Model Weights

We provide the pre-trained model weights for DualFocus.
**Download Checkpoint:** [**Baidu Netdisk**](https://pan.baidu.com/s/1o8sUfHY_zbLKkrxqdL0_Nw?pwd=ades) (Password: `ades`)

Please download the checkpoint and place it in the `models/` directory (or update the path in your config).

## 🚀 Usage

### 1. Data Preparation
Please download the standard public datasets used in the paper (MSVD-QA, MSRVTT-QA, TGIF-QA, ActivityNet-QA, Video-ChatGPT Benchmark) and organize them in the `data/` folder.

### 2. Configuration
Modify the configuration files in the `configs/` folder to match your hardware setup (e.g., number of GPUs, paths to datasets, and model checkpoints).

### 3. Training
To train the model (fine-tune on Video-ChatGPT 100K dataset), run:
```bash
python train.py --config configs/train_config.yaml
```

### 4. Evaluation
To reproduce the results in our paper:

**1.Get Video Generation:**
```bash
python eval_generate.py --config configs/eval_config.yaml
```

**2.Video Generation Evaluation:**
```bash
python eval_acc.py --config configs/eval_acc_config.yaml
```

<!-- 
## 🔗 Citation

If you find this code or paper useful, please consider citing:

```bibtex
@article{sun2024dualfocus,
  title={DualFocus: A Framework for Efficient Decoupling of Visual Representation and Focus Fusion in Video LLM},
  author={Sun, Mingyu and Zhou, Hongwei and Luo, Zhiqing and Li, Xing and Zhu, Jinhao and Yan, Tianwei and Rao, Zhibo},
  journal={Journal of LaTeX Class Files},
  year={2024}
}
```
-->

## 📄 License
This project is licensed under the Apache 2.0 License.

## 🙏 Acknowledgement
We thank the authors of [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT), [Vicuna](https://github.com/lm-sys/FastChat), and [EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP) for their open-source contributions.
