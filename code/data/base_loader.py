# fscil_cifar100.py
import random, numpy as np
from pathlib import Path
from typing import List, Dict
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from torchvision import transforms
import code.config as C
from code.data.data_utils import CutMixCollate
from code.data.miniimagenet.miniimagenet import MiniImageNet

"""
for we now replace previous CIFAR100 related or data utils to current code (for benchmark),
we would modify the base_train.py, 
incremental_run, train_biag according to this. or other model code. for 
shape should be matched. 
"""

ROOT          = Path("~/datasets").expanduser()
NUM_WORKERS   = C.NUM_WORKERS
RNG_SEED      = 2025

# for cifar100 preprocess
tf_cifar_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.507, 0.487, 0.441],
                         [0.267, 0.256, 0.276]),
])
tf_cifar_plain = transforms.Compose([                  # 증강 없이 정규화만
    transforms.ToTensor(),
    transforms.Normalize([0.507, 0.487, 0.441],
                         [0.267, 0.256, 0.276]),
])

# for preprocess Mini-ImageNet
tf_mini_train = transforms.Compose([
    transforms.RandomResizedCrop(84),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])
tf_mini_plain = transforms.Compose([
    transforms.Resize([92, 92]),
    transforms.CenterCrop(84),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

# class split for base & incremental session
# CIFAR-100
CIFAR_BASE_CLASSES   = list(range(60))                         # 0‥59
CIFAR_INCR_SESSIONS  = [list(range(60 + 5*i, 60 + 5*(i+1)))    # 60‥99
                        for i in range(8)]                      # 8 sessions

# Mini-ImageNet
MINI_BASE_CLASSES   = list(range(60))                          # 0‥59
MINI_INCR_SESSIONS  = [list(range(60 + 5*i, 60 + 5*(i+1)))     # 60‥99
                       for i in range(8)]                       # 8 sessions

def _indices_by_class(targets: List[int], wanted: List[int]) -> List[int]:
    """dataset.targets(list[int])에서 원하는 label들의 인덱스만 뽑기"""
    return [i for i, t in enumerate(targets) if t in wanted]

def build_cifar_fscil_loaders(seed: int = RNG_SEED) -> Dict[str, object]:
    """
    return dict:
      loaders['d0_train'], loaders['proto'], loaders['test_base'],
      loaders['support_sess'][s], loaders['test_sess'][s]
    """
    random.seed(seed); np.random.seed(seed)

    full_train = CIFAR100(ROOT, train=True,  download=True, transform=tf_cifar_train)
    full_test  = CIFAR100(ROOT, train=False, download=True, transform=tf_cifar_plain)

    d0_idx   = _indices_by_class(full_train.targets, CIFAR_BASE_CLASSES)
    d0_train = Subset(full_train, d0_idx)

    cutmix_collate = CutMixCollate(C.MIXUP_ALPHA, C.MIX_PROB) if C.MIX_PROB > 0 else None

    d0_loader = DataLoader(
        d0_train,
        batch_size=C.BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=cutmix_collate
    )

    # BiAG
    proto_set = CIFAR100(ROOT, train=True, transform=tf_cifar_plain)
    proto_loader = DataLoader(
        Subset(proto_set, d0_idx),
        batch_size=C.BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    d0_test = _indices_by_class(full_test.targets, CIFAR_BASE_CLASSES)
    test_loader = DataLoader(
        Subset(full_test, d0_test),
        batch_size=C.BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Incremental run
    support_loaders, test_loaders = [], []
    for sess_cls in CIFAR_INCR_SESSIONS:
        # support k-shot
        sess_train_idx = _indices_by_class(full_train.targets, sess_cls)
        support_set    = Subset(CIFAR100(ROOT, train=True, transform=tf_cifar_plain),
                                sess_train_idx)
        support_loaders.append(
            DataLoader(support_set,
                       batch_size=len(support_set),
                       shuffle=False,
                       num_workers=NUM_WORKERS,
                       pin_memory=True)
        )

        sess_test_idx = _indices_by_class(full_test.targets, sess_cls)
        test_loaders.append(
            DataLoader(Subset(full_test, sess_test_idx),
                       batch_size=C.BATCH_SIZE,
                       shuffle=False,
                       num_workers=NUM_WORKERS,
                       pin_memory=True)
        )

    return dict(
        d0_train      = d0_loader,
        proto         = proto_loader,
        d0_test       = test_loader,
        support_sess  = support_loaders,
        test_sess     = test_loaders,
        class_splits  = dict(base=CIFAR_BASE_CLASSES, sessions=CIFAR_INCR_SESSIONS),
    )

def build_mini_imagenet_fscil_loaders(seed: int = RNG_SEED) -> Dict[str, object]:
    """
    Mini-ImageNet FSCIL loaders
    return dict:
      loaders['d0_train'], loaders['proto'], loaders['test_base'],
      loaders['support_sess'][s], loaders['test_sess'][s]
    """
    random.seed(seed);
    np.random.seed(seed)

    print(f"C.DATA_ROOT:{C.DATA_ROOT}")
    mini_root = Path(C.DATA_ROOT) if C.DATA_ROOT else ROOT
    print(f"mini_root :{mini_root}")
    d0_train = MiniImageNet(
        root=mini_root,
        train=True,
        index=MINI_BASE_CLASSES,
        base_sess=True,
        do_augment=True
    )

    cutmix_collate = CutMixCollate(C.MIXUP_ALPHA, C.MIX_PROB) if C.MIX_PROB > 0 else None

    d0_loader = DataLoader(
        d0_train,
        batch_size=C.BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=cutmix_collate
    )

    proto_set = MiniImageNet(
        root=mini_root,
        train=True,
        index=MINI_BASE_CLASSES,
        base_sess=True,
        do_augment=False  # No augmentation for prototypes
    )

    proto_loader = DataLoader(
        proto_set,
        batch_size=C.BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Base-test (D0 60 classes)
    test_set = MiniImageNet(
        root=mini_root,
        train=False,
        index=MINI_BASE_CLASSES,
        base_sess=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=C.BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Incremental run
    support_loaders, test_loaders = [], []

    for session_idx, sess_cls in enumerate(MINI_INCR_SESSIONS):
        # Support set (few-shot samples)
        txt_path = mini_root / f"index_list/mini_imagenet/session_{session_idx+1}.txt"
        txt_path = str(txt_path)  # 문자열로 변환

        support_set = MiniImageNet(
            root=mini_root,
            train=True,
            index_path=txt_path,
            base_sess=False,
            do_augment=False  # No augmentation for few-shot
        )

        support_loaders.append(
            DataLoader(
                support_set,
                batch_size=len(support_set),  # Load all at once
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True
            )
        )

        # Test set (cumulative: base + all previous sessions + current)
        all_classes = MINI_BASE_CLASSES + [c for s in MINI_INCR_SESSIONS[:session_idx + 1] for c in s]
        test_set = MiniImageNet(
            root=mini_root,
            train=False,
            index=sess_cls,  # Current session classes only
            base_sess=False
        )

        test_loaders.append(
            DataLoader(
                test_set,
                batch_size=C.BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True
            )
        )

    return dict(
        d0_train=d0_loader,
        proto=proto_loader,
        d0_test=test_loader,
        support_sess=support_loaders,
        test_sess=test_loaders,
        class_splits=dict(base=MINI_BASE_CLASSES, sessions=MINI_INCR_SESSIONS),
    )


def build_fscil_loaders(dataset_name: str = None, seed: int = RNG_SEED) -> Dict[str, object]:
    """
    Unified loader builder for CIFAR-100 and Mini-ImageNet
    """
    if dataset_name is None:
        dataset_name = C.DATASET_NAME

    if dataset_name.lower() == "cifar100":
        return build_cifar_fscil_loaders(seed)
    elif dataset_name.lower() == "miniimagenet" or dataset_name.lower() == "mini_imagenet":
        return build_mini_imagenet_fscil_loaders(seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    import sys

    # Choose dataset from command line or default to config
    dataset = sys.argv[1] if len(sys.argv) > 1 else C.DATASET_NAME

    print(f"Testing {dataset} loaders...")
    loaders = build_fscil_loaders(dataset)

    print(f"D0 train batches : {len(loaders['d0_train'])}")
    print(f"Prototype loader : {len(loaders['proto'])}")
    print(f"Base classes     : {len(loaders['class_splits']['base'])}")
    print(f"Sessions         : {len(loaders['class_splits']['sessions'])}")

    # Test loading one batch
    for i, (imgs, labels) in enumerate(loaders["d0_train"]):
        print(f"Batch shape: {imgs.shape}, Labels shape: {labels.shape}")
        if isinstance(labels, tuple):  # CutMix case
            print(f"CutMix labels: {labels[0].shape}, {labels[1].shape}, lambda={labels[2]}")
        break



