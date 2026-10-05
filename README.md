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
* Windows 10 local CUDA 12.6 + python 3.12 + `torch==2.7.1+cu126`

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
| miniImageNet        | 60 classes × 500 imgs | 8 sessions × 5 classes |   5   |

---

## Getting Started

Use Python 3.12 and install the dependencies for your CPU/CUDA environment:

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The fixes and checkpoint migration are listed individually in
[reproduction_fixes.md](docs/reproduction_fixes.md) (Korean).
Use a new output directory for corrected experiments:

```bash
python main.py base --dataset cifar100 --data_root ./code/data --output_dir ./checkpoints/corrected --epochs 200 --seed 1
python main.py biag --dataset cifar100 --data_root ./code/data --output_dir ./checkpoints/corrected --biag_epochs 50 --biag_depth 4 --seed 1
python main.py incremental_run --dataset cifar100 --data_root ./code/data --output_dir ./checkpoints/corrected --biag_depth 4 --seed 1
```

The same arguments work on Windows and Linux. For miniImageNet, change
`--dataset` to `miniimagenet` and set `--data_root` to the directory containing
`miniimagenet/images` and `index_list/mini_imagenet`.
The default backbone is ResNet-18 for CIFAR-100 and ResNet-12 for miniImageNet.
Use `--biag_depth` consistently in training and evaluation. `--epochs` controls
base training; `--biag_epochs` controls generator training.

Checkpoints and `base_log.csv` / `biag_log.csv` are stored under
`<output_dir>/<dataset>/`. Evaluation writes `run/metrics.csv` and
`run/summary.json`, including separate base and novel accuracy.
The `forgetting` field now measures the drop on the same base-class test
population; it is not comparable to the historical cumulative-accuracy drop.

### Docker Usage

Build from this checkout to include these fixes. Previously published Docker
images may contain the legacy implementation.

```bash
docker build -t biag/fscil:corrected .
```

### Google Colab

[Open the notebook](https://colab.research.google.com/github/hscient/FSCIL_learn-by-reasoning/blob/main/Learn_by_reasoning.ipynb).
Use a checkout containing these fixes, including when trying the pull request
before it is merged. The notebook uses the same CLI and corrected output paths.

---

## Pre‑trained Checkpoints

The [historical CIFAR-100 release](https://github.com/hscient/FSCIL_learn-by-reasoning/releases/tag/weights-cifar100-res18-2025-07-27)
contains legacy BiAG weights. **Retrain BiAG after these fixes**: the shared SCM
architecture is incompatible with old generator checkpoints, and evaluation
rejects them explicitly. A matched base backbone/classifier/prototype bundle
can be reused for an isolated BiAG experiment. The release does not include a
prototype file; for a complete fresh run, use the commands above.

---

## Benchmark Results

The corrected implementation has **not yet been benchmarked after full training**.
Historical results below are preserved for reference, not attributed to the fixes.

| CIFAR-100 | Session 0 | Final session | Average across sessions |
| --- | ---: | ---: | ---: |
| Paper, Table III | 84.00 | 57.95 | 68.93 |
| Historical repository report | 82.92 | 49.74 | Unverified |
| Corrected implementation | Not measured | Not measured | Not measured |

The historical final-session gap is **8.21 percentage points**. `68.93` is
the paper's session average, not final accuracy. The old README also listed
`63.88` under a misleading heading; its aggregation cannot be verified from
the committed experiment logs and is not treated as a paper result.

## Issue Tracker & TODO

* [x] Correct episodic loss, SCM connections, seen-class inference and CutMix labels.
* [x] Add CPU regression tests and base/novel evaluation metrics.
* [ ] Retrain on the complete benchmarks and publish session-wise logs.
* [ ] Measure the effects of individual changes with controlled ablations.
* [ ] CUB-200 dataloader and configs.

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

---

## License

This implementation is released under the **MIT License**. See [LICENSE](LICENSE) for the full text.

The original BiAG paper, its figures and tables are © the respective authors.

Parts of the dataloader were adapted from the official implementation of CEC

---

## Citation

> **Disclaimer:** This repository is an **unofficial implementation** of *Learn by Reasoning: Analogical Weight Generation for Few‑Shot Class‑Incremental Learning*. It is under active development; hyperparameters and training schedules are still being tuned, so results may differ from the paper.

If you use this code or find it helpful, please cite the original paper:

```bibtex
@misc{han2025learn,
  title={Learn by Reasoning: Analogical Weight Generation for Few-Shot Class-Incremental Learning},
  author={Jizhou Han and Chenhao Ding and Yuhang He and Songlin Dong and Qiang Wang and Xinyuan Gao and Yihong Gong},
  eprint={2503.21258},
  archivePrefix={arXiv},
  year={2025}
}
```
