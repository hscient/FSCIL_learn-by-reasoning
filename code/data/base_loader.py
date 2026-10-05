"""Base training/prototype loaders; incremental loaders use fixed session files."""
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from torchvision import transforms
import code.config as C
from code.data.data_utils import CutMixCollate
from code.data.miniimagenet.miniimagenet import MiniImageNet
from code.utils.utils import seed_everything

CIFAR_BASE_CLASSES = list(range(60))
MINI_BASE_CLASSES = list(range(60))
tf_cifar_plain = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.507, 0.487, 0.441], [0.267, 0.256, 0.276]),
])
tf_cifar_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
    *tf_cifar_plain.transforms,
])


def _loader(dataset, train=False):
    collate = None
    if train and C.USE_CUTMIX and C.MIX_PROB > 0:
        collate = CutMixCollate(C.MIXUP_ALPHA, C.MIX_PROB)
    return DataLoader(dataset, batch_size=C.BATCH_SIZE, shuffle=train,
                      num_workers=C.NUM_WORKERS, pin_memory=True, collate_fn=collate)


def build_cifar_fscil_loaders(seed=2025):
    seed_everything(seed)
    root = Path(C.DATA_ROOT).expanduser()
    train = CIFAR100(root, train=True, download=True, transform=tf_cifar_train)
    proto = CIFAR100(root, train=True, download=False, transform=tf_cifar_plain)
    test = CIFAR100(root, train=False, download=True, transform=tf_cifar_plain)

    def base_subset(dataset):
        return Subset(dataset, [i for i, y in enumerate(dataset.targets)
                                if y in CIFAR_BASE_CLASSES])

    return dict(d0_train=_loader(base_subset(train), train=True),
                proto=_loader(base_subset(proto)), d0_test=_loader(base_subset(test)),
                class_splits=dict(base=CIFAR_BASE_CLASSES))


def build_mini_imagenet_fscil_loaders(seed=2025):
    seed_everything(seed)
    root = Path(C.DATA_ROOT).expanduser()
    train = MiniImageNet(root=root, train=True, index=MINI_BASE_CLASSES,
                         base_sess=True, do_augment=True)
    proto = MiniImageNet(root=root, train=True, index=MINI_BASE_CLASSES,
                         base_sess=True, do_augment=False)
    test = MiniImageNet(root=root, train=False, index=MINI_BASE_CLASSES,
                        base_sess=True, do_augment=False)
    return dict(d0_train=_loader(train, train=True), proto=_loader(proto),
                d0_test=_loader(test), class_splits=dict(base=MINI_BASE_CLASSES))


def build_fscil_loaders(dataset_name=None, seed=2025):
    dataset_name = (dataset_name or C.DATASET_NAME).lower()
    if dataset_name == "cifar100":
        return build_cifar_fscil_loaders(seed)
    if dataset_name in ("miniimagenet", "mini_imagenet"):
        return build_mini_imagenet_fscil_loaders(seed)
    raise ValueError(f"Unknown dataset: {dataset_name}")
