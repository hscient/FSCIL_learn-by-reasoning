import sys
import json

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

}

def update_from_args(ns: dict):
    for k, v in ns.items():
        if v is None:
            continue
        key = CLI2CFG.get(k, k.upper())
        if key in globals():
            globals()[key] = v

    global IMAGE_SIZE, DATASET_NAME

def effective_config_dict():
    return {
        k: v for k, v in globals().items()
        if k.isupper() and not k.startswith("_")
    }

def print_effective_config(stream=sys.stdout):
    cfg = effective_config_dict()
    maxk = max(len(k) for k in cfg.keys())
    print("=== CONFIG ===", file=stream)
    for k in sorted(cfg.keys()):
        print(f"{k:<{maxk}} = {cfg[k]}", file=stream)

def save_effective_config(path: str):
    with open(path, "w") as f:
        json.dump(effective_config_dict(), f, indent=2, default=str)
    print(f"[config] effective config saved to {path}")
