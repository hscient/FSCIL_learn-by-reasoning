# Learn by Reasoning: Analogical Weight Generation for Few‑Shot Class‑Incremental Learning (BiAG)

> This is **not an official** implementation.
> **“Learn by Reasoning: Analogical Weight Generation for Few‑Shot Class‑Incremental Learning”** ([ArXiv 2503.21258](https://arxiv.org/abs/2503.21258)).

---

## Table of Contents

1. [Environment](#Environment)
2. [Docker Usage](#docker-usage)
3. [Data Preparation](#data-preparation)
4. [Getting Started](#getting-started)
5. [Pre‑trained Checkpoints](#pre-trained-checkpoints)
6. [Benchmark Results](#benchmark-results)
7. [Issue Tracker & TODO](#issue-tracker--todo)
8. [Acknowledgment](#acknowledgment)
9. [Colab Sharing](#colab-sharing)
10. [License](#license)
11. [Citation](#citation)

---

## Environment

### Tested platforms

* Google Colab (T4 / A100)
* Ubuntu 22.04 + CUDA 11.8
* Windows 10 local CUDA 12.6 + `torch==2.7.1+cu126`

---

## Data Preparation

* **CIFAR‑100** is downloaded automatically via `torchvision`.
* For **miniImageNet**, use the data link provided by the
  [CEC‑CVPR2021 repository](https://github.com/icoz69/CEC-CVPR2021?tab=readme-ov-file);
  the repo gives a download link **[here](https://drive.google.com/drive/folders/11LxZCQj2FRCs0JTsf_dafvTHqFn2yGSN)**.
  you can download the dataset and unzip it under code/data folder

> **Note**: CIFAR‑100 & miniImageNet follows the **CEC** split (60 base + 40 novel). 

**Session configuration (examples):**

| Dataset             | Base session          | #Incremental sessions  | Shots |
| ------------------- |-----------------------|------------------------| :---: |
| CIFAR‑100           | 60 classes × 500 imgs | 8 sessions × 5 classes |   5   |
| miniImageNet        | 60 classes × 600 imgs | 9 sessions × 5 classes |   5   |

---

## Getting Started

the code was tested three execution environments : **Linux Ubuntu Docker**, **Google Colab**, and **Local (Windows)**.

### 1) Docker 

Ready‑to‑use images are published on Docker Hub.

1. Quick start

| Action                 | Command                                 | When to use                                              |
| ---------------------- | --------------------------------------- | -------------------------------------------------------- |
| **Pull** (recommended) | `docker pull airiter/biag-fscil:latest` | just want to *use* the code – faster start‑up, reproducible. |
| **Build**              | `docker build -t biag/fscil:latest .`   | modified the source or need a custom CUDA/cuDNN base.    |


2. base train example

```bash
docker run --rm -it -v your_code_path -w /app airiter/biag-fscil:latest \
python main.py base \
  --dataset cifar100 \
  --epochs 500 \
  --backbone_model resnet18 \
  --batch_size 128 \
  --pt_backbone checkpoints/cifar100/backbone_18.pt \
  --pt_classifier checkpoints/cifar100/classifier_18.pt \
  --pt_proto checkpoints/cifar100/prototype_18.pt \
  --num_workers 4 \
  --show_config
```

3. biag train example
```bash
docker run --rm -it -v your_code_path -w /app airiter/biag-fscil:latest \
python main.py biag \
  --dataset cifar100 \
  --backbone_model resnet18 \
  --batch_size 64 \
  --pt_biag_last checkpoints/cifar100/biag_18.pt \
  --num_workers 4 \
  --show_config
```
4. incremental run example
```bash
docker run --rm -it -v your_code_path -w /app airiter/biag-fscil:latest \
python main.py incremental_run \
  --dataset cifar100 \
  --pt_backbone checkpoints/cifar100/backbone_18.pt \
  --pt_classifier checkpoints/cifar100/classifier_18.pt \
  --pt_proto checkpoints/cifar100/prototype_18.pt \
  --biag checkpoints/cifar100/biag_18.pt \
  --num_workers 4 \
  --show_config
```

---

### 2) Google Colab (One‑click)

Open the notebook in your browser—no local setup required.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/hscient/FSCIL_learn-by-reasoning/blob/main/Learn_by_reasoning.ipynb)

[Open in Colab](https://colab.research.google.com/github/hscient/FSCIL_learn-by-reasoning/blob/main/Learn_by_reasoning.ipynb)

The Colab notebook installs dependencies, downloads datasets (or mounts Drive), and exposes the same CLI commands as below.
for miniImageNet dataset, you should mount dataset to google drive

---

### 3) Local (Windows)

**Tested:** Python 3.12, CUDA 12.6, requirement_local.txt includes all library version tested

**A. Base Module Training (CIFAR‑100)**

```bash
python main.py base \
  --dataset cifar100 \
  --epochs 500 \
  --backbone_model resnet18 \
  --batch_size 128 \
  --pt_backbone checkpoints/cifar100/backbone_18.pt \
  --pt_classifier checkpoints/cifar100/classifier_18.pt \
  --pt_proto checkpoints/cifar100/prototype_18.pt \
  --num_workers 2 \
  --show_config \
  --save_config ./cfg_after_cli.json
```

**B. BiAG Module Training**

```bash
python main.py biag \
  --dataset cifar100 \
  --epochs 50 \
  --batch_size 64 \
  --num_workers 4 \
  --pt_backbone checkpoints/cifar100/backbone_18.pt \
  --pt_classifier checkpoints/cifar100/classifier_18.pt \
  --pt_proto checkpoints/cifar100/prototype_18.pt \
  --biag checkpoints/cifar100/biag_18_last.pt \
  --show_config
```

**C. Incremental Learning Evaluation**

```bash
python main.py incremental_run \
  --dataset cifar100 \
  --backbone_model resnet18 \
  --biag checkpoints/cifar100/biag_18_last.pt \
  --pt_backbone checkpoints/cifar100/backbone_18.pt \
  --pt_classifier checkpoints/cifar100/classifier_18.pt \
  --pt_proto checkpoints/cifar100/prototype_18.pt
```

**Notes**

* **Logs & Checkpoints**

  * Logs are saved under `logs/`.

---

## Pre‑trained Checkpoints

> **CIFAR‑100**

| Dataset      | Backbone  | Download                                                                                               |
| ------------ |-----------| ------------------------------------------------------------------------------------------------------ |
| CIFAR‑100    | ResNet‑18 | [`biag_cifar100_res12.pt`](https://github.com/your-repo/releases/download/v0.1/biag_cifar100_res12.pt) |

---

## Benchmark Results

> **Note:** We are still hyper‑parameter tuning – results are **below** paper numbers.

**CIFAR‑100 (paper):**

| Dataset      | Session‑0 | Final session | Paper (final) |
| ------------ |:---------:|:-------------:|:-------------:|
| CIFAR‑100    |   84.00   |     57.95     |     68.93     |

**CIFAR‑100 (this repo, WIP):**

| Dataset      | Session‑0 | Final session | Paper (final) |
| ------------ |:---------:|:-------------:|:------------:|
| CIFAR‑100    |           |               |              |

Log files can be found under `logs/`.

---

Planned tasks:

* [ ] CUB‑200 dataloader & configs
* [ ] Hyper‑param sweep → close paper gap

---

## Acknowledgment

Our code is based on:

* **FSCIL (Dataset):** [https://github.com/xyutao/fscil](https://github.com/xyutao/fscil)
* **CEC (Dataloader):** [https://github.com/icoz69/CEC-CVPR2021](https://github.com/icoz69/CEC-CVPR2021)

---

## Colab Usage

Following colab notebook's instruction as run the colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/hscient/FSCIL_learn-by-reasoning/blob/main/Learn_by_reasoning.ipynb)
[Open in Colab](https://colab.research.google.com/github/hscient/FSCIL_learn-by-reasoning/blob/main/Learn_by_reasoning.ipynb)

---

## License

This implementation is released under the **MIT License**. See [LICENSE](LICENSE) for the full text.

The original BiAG paper, its figures and tables are © the respective authors.

Parts of the dataloader were adapted from the official implementation of **Constrained Few‑shot Class‑Incremental Learning**.

---

## Citation

> **Disclaimer:** This repository is an **unofficial implementation** of *Learn by Reasoning: Analogical Weight Generation for Few‑Shot Class‑Incremental Learning*. It is under active development; hyperparameters and training schedules are still being tuned, so results may differ from the paper.

If you use this code or find it helpful, please cite the original paper:

```bibtex
@inproceedings{han2025learn,
  title={Learn by Reasoning: Analogical Weight Generation for Few-Shot Class-Incremental Learning},
  author={Jizhou Han and Chenhao Ding and Yuhang He and Songlin Dong and Qiang Wang and Xinyuan Gao and Yihong Gong},
  booktitle={CVPR},
  year={2025}
}
```
