# DualFocus: A Framework for Efficient Decoupling of Visual Representation and Focus Fusion in Video LLM

> **Note:** This repository contains the official implementation for the IEEE paper submission titled "DualFocus: A Framework for Efficient Decoupling of Visual Representation and Focus Fusion in Video LLM".

## Blind Review Notice

This code has been released for the purpose of blind review. To maintain anonymity, the repository and its contents have been stripped of any identifying information about the authors or their institutions. We kindly request that reviewers maintain this anonymity during the review process.

---

The model checkpoint is Here:
[**Download from Drive**](https://pan.baidu.com/s/1o8sUfHY_zbLKkrxqdL0_Nw?pwd=ades).
You can completely reproduce the results in our paper and analyze it.

---

Here are the setup and execution steps:

1.  **Environment Setup:** Create a Conda virtual environment and install the necessary dependencies listed in `requirements.txt`. Ensure that `accelerate` and `deepspeed` are properly configured within this environment.
2.  **Data and Models:** Download the base models (as mentioned previously) into the `models/` folder. Additionally, download the standard public datasets used in the paper.
3.  **Configuration:** Configure the relevant files in the `configs/` folder according to your specific environment and hardware setup.
4.  **Execution:** Finally, run `train.py`, `eval_generate.py`, and `eval_acc.py` from your terminal.
