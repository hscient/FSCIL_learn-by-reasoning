# -*- coding: utf-8 -*-
"""fscil_loader_legacy.py  ‑ patched
------------------------------------------------------------
Loads CIFAR‑100 episodes exactly like IBM‑FSCIL (session_*.txt index files)
so that incremental_run.py can compare apples‑to‑apples with prior work.

Fixes applied
~~~~~~~~~~~~~
1. **NumPy ≥ 1.24 bug** in ``SelectfromDefault`` is patched *before* the first
   dataset instantiation → prevents the *broadcast together* error.
2. Import paths updated for this project layout (``dataloader.cifar100``).
3. Uses *session_n.txt* files for support loaders & cumulative test loaders.

Returned dict keys are unchanged: ``d0_train``, ``proto``, ``d0_test``,
``support_sess``, ``test_sess``, ``class_splits``.
"""
from __future__ import annotations

import importlib, random, numpy as np
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader
from code.data import data_utils as du
from code import config as C

__all__ = ["build_cifar_fscil_loaders", "build_mini_imagenet_loaders"]


def _patch_selectfromdefault_if_needed() -> None:
    """Replace buggy equality check with len() guard (idempotent)."""
    candidate_paths = [
        "code.data.cifar100.cifar100",
    ]
    for mod_name in candidate_paths:
        try:
            cifar_mod = importlib.import_module(mod_name)
            break
        except ModuleNotFoundError:
            cifar_mod = None

    if cifar_mod is None:
        return

    def _safe_select(self, data, targets, index):
        """NumPy ≥1.24 호환 · 빈 배열 비교 제거"""
        data_chunks, target_chunks = [], []
        for cls in index:
            mask = targets == cls
            if mask.any():
                data_chunks.append(data[mask])
                target_chunks.append(targets[mask])
        if not data_chunks:                         # if all classes are empty
            raise ValueError(f"No samples for classes {index}")
        return np.concatenate(data_chunks), np.concatenate(target_chunks)

    def _safe_newclass(self, data, targets, index):
        ind_np = np.asarray(index, dtype=int)
        if ind_np.size == 0:
            raise ValueError("Empty index list for new‑class selector")
        return data[ind_np], targets[ind_np]

    for cls_name in ("cifar10", "cifar100", "CIFAR10", "CIFAR100"):
        if hasattr(cifar_mod, cls_name):
            cls_obj = getattr(cifar_mod, cls_name)
            cls_obj.SelectfromDefault = _safe_select
            cls_obj.NewClassSelector  = _safe_newclass

# def _make_args(way: int, shot: int, query: int) -> SimpleNamespace:
#     """Create an args namespace exactly like IBM code expects."""
#     args = SimpleNamespace(
#         dataset="cifar100",
#         data_folder=str(Path(C.DATA_ROOT).expanduser()),
#         base_class=C.BASE_CLASS,
#         num_classes=C.NUM_CLASSES,
#         sessions=C.SESSIONS,
#         way=way,
#         shot=shot,
#         batch_size_training=C.BATCH_INCREMENTAL_SIZE,
#         batch_size_inference=C.BATCH_SIZE,
#         num_workers=8,
#         num_ways_training=way,
#         num_shots_training=shot,
#         num_query_training=query,
#         max_train_iter=C.PSEUDO_EPISODES_PER_EPOCH,
#     )
#
#     # Patch BEFORE instantiating any dataset!
#     _patch_selectfromdefault_if_needed()
#
#     # Attach dataset wrappers (CIFAR100 with SelectfromDefault call inside)
#     args = du.set_up_datasets(args)
#     return args

def _make_args(way: int, shot: int, query: int, dataset: str = None) -> SimpleNamespace:
    """Create an args namespace exactly like IBM code expects."""
    dataset = (dataset or getattr(C, "DATASET_NAME", "cifar100")).lower()
    args = SimpleNamespace(
        dataset=dataset,
        data_folder=str(Path(C.DATA_ROOT).expanduser()),
        base_class=C.BASE_CLASS,
        num_classes=C.NUM_CLASSES,
        sessions=C.SESSIONS,
        way=way,
        shot=shot,
        batch_size_training=C.BATCH_INCREMENTAL_SIZE,
        batch_size_inference=C.BATCH_SIZE,
        num_workers=8,
        num_ways_training=way,
        num_shots_training=shot,
        num_query_training=query,
        max_train_iter=C.PSEUDO_EPISODES_PER_EPOCH,
    )

    _patch_selectfromdefault_if_needed()  # safe for miniimagenet도 무해
    args = du.set_up_datasets(args)
    return args

# def build_cifar_fscil_loaders(
#     seed: int = C.SEED,
#     way: Optional[int] = None,
#     shot: Optional[int] = None,
#     query: Optional[int] = None,
# ) -> Dict[str, object]:
#     """Return loaders that rely on *session_n.txt* index files."""
#     way = way or C.WAY
#     shot = shot or C.SHOT
#     query = query or getattr(C, "QUERY", 15)
#
#     random.seed(seed);
#     np.random.seed(seed);
#     torch.manual_seed(seed)
#
#     args = _make_args(way, shot, query)
def build_cifar_fscil_loaders(seed: int = C.SEED, way=None, shot=None, query=None):
    way = way or C.WAY;
    shot = shot or C.SHOT;
    query = query or getattr(C, "QUERY", 15)
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    args = _make_args(way, shot, query, dataset="cifar100")

    _, _, d0_test_loader = du.get_base_dataloader(args)

    support_loaders: List[DataLoader] = []
    test_loaders:    List[DataLoader] = []
    session_classes: List[List[int]]  = []

    for s in range(1, args.sessions):
        train_set, train_loader, test_loader = du.get_new_dataloader(
            args, session=s, do_augment=False
        )
        support_loaders.append(train_loader)
        test_loaders.append(test_loader)
        session_classes.append([int(c) for c in du.get_session_classes(args, s)])

    return dict(
        d0_test      = d0_test_loader,          # base model accuracy 평가용
        support_sess = support_loaders,         # 각 세션 5‑shot support
        test_sess    = test_loaders,            # 누적 test 셋
        class_splits = dict(
            base=list(range(args.base_class)),
            sessions=session_classes,
        ),
    )

# def build_mini_imagenet_loaders(
#     seed: int = C.SEED,
#     way: Optional[int] = None,
#     shot: Optional[int] = None,
#     query: Optional[int] = None,
# ) -> Dict[str, object]:
#     """Return loaders that rely on *session_n.txt* index files."""
#     way = way or C.WAY
#     shot = shot or C.SHOT
#     query = query or getattr(C, "QUERY", 15)
#
#     random.seed(seed);
#     np.random.seed(seed);
#     torch.manual_seed(seed)
#
#     args = _make_args(way, shot, query)

def build_mini_imagenet_loaders(seed: int = C.SEED, way=None, shot=None, query=None):
    way = way or C.WAY;
    shot = shot or C.SHOT;
    query = query or getattr(C, "QUERY", 15)
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    args = _make_args(way, shot, query, dataset="miniimagenet")

    _, _, d0_test_loader = du.get_base_dataloader(args)

    support_loaders: List[DataLoader] = []
    test_loaders:    List[DataLoader] = []
    session_classes: List[List[int]]  = []

    for s in range(1, args.sessions):
        train_set, train_loader, test_loader = du.get_new_dataloader(
            args, session=s, do_augment=False
        )
        support_loaders.append(train_loader)
        test_loaders.append(test_loader)
        session_classes.append([int(c) for c in du.get_session_classes(args, s)])

    return dict(
        d0_test      = d0_test_loader,          # base model accuracy 평가용
        support_sess = support_loaders,         # 각 세션 5‑shot support
        test_sess    = test_loaders,            # 누적 test 셋
        class_splits = dict(
            base=list(range(args.base_class)),
            sessions=session_classes,
        ),
    )