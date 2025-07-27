import sys, json
from typing import Dict, Iterable, Set


NUM_WORKERS = 4

# ╭──────────────── DATASET ───────────────╮
DATASET_NAME             = "cifar100"
NUM_CLASSES              = 100            # total classes across all sessions
IMAGE_SIZE               = 32             # default; will be set dynamically below

# Dataset-specific image sizes
MINI_IMAGE_SIZE         = 84
CIFAR100_IMAGE_SIZE     = 32
BACKBONE_MODEL          = "resnet18"     # ["resnet12", "resnet18", …]

# Update IMAGE_SIZE based on dataset
if DATASET_NAME.lower() == "miniimagenet" or DATASET_NAME.lower() == "mini_imagenet":
    IMAGE_SIZE = MINI_IMAGE_SIZE
    BACKBONE_MODEL = "resnet12"
elif DATASET_NAME.lower() == "cifar100":
    IMAGE_SIZE = CIFAR100_IMAGE_SIZE

DATA_ROOT                = "/content/drive/MyDrive/Learn-by-Reasoning/code/data"  # miniimagenet directory (miniimagenet tar file) is located
# DATA_ROOT                = "C:/Users/0dudd/PycharmProjects/Learn-by-Reasoning/code/data"

# ╭──────────────── TRAINING (BASE) ───────╮

EPOCHS                   = 200
INIT_LR                  = 0.1
BATCH_SIZE               = 128
BATCH_INCREMENTAL_SIZE   = 64             # mini-batch when classifier expands

# Augmentation switches
MIXUP_ALPHA              = 1.0            
MIX_PROB = 0.5

# ╭──────────────── BIAG (ANALOGICAL GEN) ─╮
BIAG_EPOCHS              = 50
BIAG_DEPTH               = 4              # num Transformer-like layers
BIAG_LR                  = 1e-3
BIAG_OPTIMIZER           = "AdamW"  #"SGD"        # ["AdamW", "SGD"]
PSEUDO_EPISODES_PER_EPOCH= 200            # generates episodic tasks on-the-fly

# ╭──────────────── FSCIL SETTINGS ────────╮
INIT_K_SHOT              = 5              # default 5-shot benchmark
SHOT_PER_CLASS           = INIT_K_SHOT    # alias (easy to override in scripts)

# Dataset-specific settings
if DATASET_NAME.lower() == "miniimagenet" or DATASET_NAME.lower() == "mini_imagenet":
    BASE_CLASS = 60
    NUM_CLASSES = 100
    WAY = 5
    SHOT = 5
    SESSIONS = 9
elif DATASET_NAME.lower() == "cifar100":
    BASE_CLASS = 60
    NUM_CLASSES = 100
    WAY = 5
    SHOT = 5
    SESSIONS = 8

# evaluation detail level
EVAL_TOP1_ONLY           = False
EVAL_METRICS             = ["top1",
                             "classwise_acc",
                             "forgetting"] # add/remove as desired

# ╭──────────────── PATHS ─────────────────╮
CHECKPOINT_DIR           = "../checkpoints"
LOG_DIR                  = "logs"

# ╭──────────────── MISC ──────────────────╮
SEED                     = 42
DEVICE                   = "cuda:0"


CLI2CFG = {
    "dataset": "DATASET_NAME",
    "data_root": "DATA_ROOT",
    "epochs": "EPOCHS",
    "init_lr": "INIT_LR",
    "backbone_model": "BACKBONE_MODEL",
    "batch_size": "BATCH_SIZE",
    "seed": "SEED",
    "biag_epochs": "BIAG_EPOCHS",
    "biag_lr": "BIAG_LR",
    "biag_depth": "BIAG_DEPTH",
    "num_workers": "NUM_WORKERS",
}

_INTERNAL_KEYS: Set[str] = {
    "CLI2CFG", "CONFIG_GROUPS", "ENTRYPOINT_GROUPS", "DEFAULT_SNAPSHOT"
}

def update_from_args(ns: dict):
    for k, v in ns.items():
        if v is None:
            continue
        key = CLI2CFG.get(k, k.upper())
        if key in globals():
            globals()[key] = v

    global IMAGE_SIZE, DATASET_NAME
#
# def effective_config_dict():
#     return {
#         k: v for k, v in globals().items()
#         if k.isupper() and not k.startswith("_")
#     }
#
# def print_effective_config(stream=sys.stdout):
#     cfg = effective_config_dict()
#     maxk = max(len(k) for k in cfg.keys())
#     print("=== CONFIG ===", file=stream)
#     for k in sorted(cfg.keys()):
#         print(f"{k:<{maxk}} = {cfg[k]}", file=stream)
#
# def save_effective_config(path: str):
#     with open(path, "w") as f:
#         json.dump(effective_config_dict(), f, indent=2, default=str)
#     print(f"[config] effective config saved to {path}")




# 출력에서 항상 숨길 내부 키들
_INTERNAL_KEYS: Set[str] = {
    "CLI2CFG", "CONFIG_GROUPS", "ENTRYPOINT_GROUPS", "DEFAULT_SNAPSHOT"
}

# 섹션(그룹) 정의: 필요 시 여기만 수정하면 유지보수 쉬움
CONFIG_GROUPS: Dict[str, Set[str]] = {
    # 헤더 주석과 1:1 대응
    "dataset": {
        "DATASET_NAME", "NUM_CLASSES", "IMAGE_SIZE",
        "MINI_IMAGE_SIZE", "CIFAR100_IMAGE_SIZE", "DATA_ROOT",
    },
    "base": {
        "EPOCHS", "INIT_LR", "BATCH_SIZE", "NUM_WORKERS",
        "BACKBONE_MODEL", "MIXUP_ALPHA", "MIX_PROB", "SEED",
    },
    "biag": {
        "BIAG_EPOCHS", "BIAG_DEPTH", "BIAG_LR", "BIAG_OPTIMIZER",
        "PSEUDO_EPISODES_PER_EPOCH",
    },
    "fscil": {
        "INIT_K_SHOT", "SHOT_PER_CLASS", "BASE_CLASS", "WAY", "SHOT", "SESSIONS",
    },
    "eval": {
        "EVAL_TOP1_ONLY", "EVAL_METRICS",
    },
    "paths": {
        "CHECKPOINT_DIR", "LOG_DIR",
    },
    "misc": {
        "DEVICE",
    },
}

# 엔트리포인트(실행 스텝) → 기본 표시 섹션 구성
ENTRYPOINT_GROUPS: Dict[str, Iterable[str]] = {
    "base":        ("dataset", "base", "fscil", "eval", "paths", "misc"),
    "biag":        ("dataset", "biag", "fscil", "eval", "paths", "misc"),
    "incremental": ("dataset", "base", "biag", "fscil", "eval", "paths", "misc"),
}

def _is_exportable_key(k: str) -> bool:
    return k.isupper() and not k.startswith("_") and k not in _INTERNAL_KEYS

def effective_config_dict(include_internals: bool = False) -> dict:
    cfg = {k: v for k, v in globals().items() if _is_exportable_key(k)}
    if include_internals:
        # 개발용으로 내부 키까지 보고 싶을 때만 True
        cfg.update({k: globals()[k] for k in _INTERNAL_KEYS if k in globals()})
    return cfg

DEFAULT_SNAPSHOT = {k: v for k, v in effective_config_dict().items()}

def _select_by_groups(cfg: dict, groups: Iterable[str] | None) -> dict:
    if not groups:
        return cfg
    allow: Set[str] = set()
    for g in groups:
        allow |= set(CONFIG_GROUPS.get(g, ()))
    return {k: v for k, v in cfg.items() if k in allow}

def print_effective_config(
    stream=sys.stdout,
    groups: Iterable[str] | None = None,
    only_changed: bool = False,
    group_headers: bool = True,
):
    """
    groups: ("dataset","base",...) section name list
    only_changed: print changed value only
    group_headers: print section header
    """
    full = effective_config_dict()
    if only_changed:
        full = {k: v for k, v in full.items() if DEFAULT_SNAPSHOT.get(k) != v}

    if groups:
        ordered_groups = list(groups)
        print("=== CONFIG ===", file=stream)
        for g in ordered_groups:
            keys = CONFIG_GROUPS.get(g, ())
            sect = {k: full[k] for k in sorted(keys) if k in full}
            if not sect:
                continue
            if group_headers:
                print(f"\n# [{g.upper()}]", file=stream)
            maxk = max(len(k) for k in sect) if sect else 0
            for k in sorted(sect):
                print(f"{k:<{maxk}} = {sect[k]}", file=stream)
    else:
        print("=== CONFIG ===", file=stream)
        if not full:
            print("(no keys to show)", file=stream)
            return
        maxk = max(len(k) for k in full)
        for k in sorted(full):
            print(f"{k:<{maxk}} = {full[k]}", file=stream)

def save_effective_config(path: str, groups: Iterable[str] | None = None, only_changed: bool = False):
    full = effective_config_dict()
    if only_changed:
        full = {k: v for k, v in full.items() if DEFAULT_SNAPSHOT.get(k) != v}
    full = _select_by_groups(full, groups)
    with open(path, "w") as f:
        json.dump(full, f, indent=2, default=str)
    print(f"[config] effective config saved to {path}")

def print_config_for(entrypoint: str, **kwargs):
    """
    entrypoint: 'base' | 'biag' | 'incremental' etc.
    kwargs is same with print_effective_config (only_changed, stream etc)
    """
    groups = ENTRYPOINT_GROUPS.get(entrypoint.lower())
    if not groups:
        return print_effective_config(**kwargs)
    return print_effective_config(groups=groups, **kwargs)
