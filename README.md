# Learn by Reasoning: Analogical Weight Generation for Few‑Shot Class‑Incremental Learning (BiAG)

> This is **not an official** implementation.
> **“Learn by Reasoning: Analogical Weight Generation for Few‑Shot Class‑Incremental Learning”** ([ArXiv 2503.21258](https://arxiv.org/abs/2503.21258)).

---

## Table of Contents

1. [Environment](#environment)
2. [Data Preparation](#data-preparation)
3. [Getting Started](#getting-started)
4. [Pre‑trained Checkpoints](#pre-trained-checkpoints)
5. [Benchmark Results](#benchmark-results)
6. [Docker Usage](#docker-usage)
7. [Issue Tracker & TODO](#issue-tracker--todo)
8. [Acknowledgment](#acknowledgment)
9. [Colab Sharing](#colab-sharing)
10. [License](#license)
11. [Citation](#citation)

---

## Environment

> ### Quick start (Docker)

Prebuilt Docker image on Docker Hub:

1. Pull the image

```bash
docker pull airiter/biag-fscil:latest
```

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

### Tested platforms

* **Google Colab** (T4 / A100)
* Ubuntu 22.04 + CUDA 11.8
* Windows 10 local CUDA 12.6 + `torch==2.7.1+cu126`

---

## Data Preparation

* **CIFAR‑100** is downloaded automatically via `torchvision`.
* For **miniImageNet**, use the data link provided by the
  [CEC‑CVPR2021 repository](https://github.com/icoz69/CEC-CVPR2021?tab=readme-ov-file);
  the repo gives a download link **[here](https://drive.google.com/drive/folders/11LxZCQj2FRCs0JTsf_dafvTHqFn2yGSN)**.
  you can download the dataset and unzip it under code/data folder

> **Note**: miniImageNet follows the **CEC** split (60 base + 40 novel). Make sure your version matches.

**Session configuration (examples):**

| Dataset             | Base session          | #Incremental sessions  | Shots |
| ------------------- |-----------------------|------------------------| :---: |
| CIFAR‑100           | 60 classes × 500 imgs | 8 sessions × 5 classes |   5   |
| miniImageNet        | 60 classes × 600 imgs | 9 sessions × 5 classes |   5   |

---

## Getting Started

the code was tested three execution environments : **Linux Ubuntu Docker**, **Google Colab**, and **Local (Windows)**.

### 1) Docker 

Ensure Docker is installed and the NVIDIA runtime is available if you plan to use GPUs.

```bash
# Pull a prebuilt image
docker pull airiter/biag-fscil:latest

# See CLI options (CPU dry run)
docker run --rm -it airiter/biag-fscil:latest python main.py --help

# Example: train base module on CIFAR-100 with GPU
docker run --gpus all --rm -it \
  -v $PWD:/workspace -w /workspace \
  airiter/biag-fscil:latest \
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

**Checkpoint defaults (when not specified):**

* `--pt_backbone`: `checkpoints/<dataset>/backbone_pt_last.pt`
* `--pt_classifier`: `checkpoints/<dataset>/classifier_pt_last.pt`
* `--pt_proto`: `checkpoints/<dataset>/proto_pt_last.pt`

---

### 2) Google Colab (One‑click)

Open the notebook in your browser—no local setup required.

* **Open in Colab:**
  [Open in Colab](https://colab.research.google.com/github/<USER>/<REPO>/blob/main/notebooks/biag_colab.ipynb)

* **Badge for this README:**

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/<USER>/<REPO>/blob/main/notebooks/biag_colab.ipynb)
```

The Colab notebook installs dependencies, downloads datasets (or mounts Drive), and exposes the same CLI commands as below.

---

### 3) Local (Windows)

**Tested:** Python 3.12, CUDA 12.6, requirement_local.txt library versions

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

* **Datasets**

  * CIFAR‑100 is downloaded automatically via `torchvision`.
  * miniImageNet: download from the CEC repository (see **Datasets and pretrained models** 
  
* **Logs & Checkpoints**

  * Logs are saved under `logs/`.
  * Pre‑trained checkpoints can be placed under `${LOGLOC}` and used with `--resume-from` for evaluation‑only runs.

---

## Pre‑trained Checkpoints

> I've test Cifar100 with paper

| Dataset      | Backbone  | Download                                                                                               |
| ------------ | --------- | ------------------------------------------------------------------------------------------------------ |
| CIFAR‑100    | ResNet‑12 | [`biag_cifar100_res12.pt`](https://github.com/your-repo/releases/download/v0.1/biag_cifar100_res12.pt) |

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

## Docker Usage

Ready‑to‑use images are published on **Docker Hub**.

| Action                 | Command                                 | When to use                                                      |
| ---------------------- | --------------------------------------- | ---------------------------------------------------------------- |
| **Pull** (recommended) | `docker pull airiter/biag-fscil:latest` | You just want to *use* the code – faster start‑up, reproducible. |
| **Build**              | `docker build -t biag/fscil:latest .`   | You modified the source or need a custom CUDA/cuDNN base.        |

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
