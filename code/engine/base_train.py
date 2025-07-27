"""Joint training loop that learns *both* backbone and classifier together.

After training, it returns the separately‑trained backbone and classifier
modules plus the base‑session prototypes so that the downstream FSCIL
pipeline (incremental sessions) can keep working unchanged.
"""
from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import code.config as C
from ..utils.logger import CSVLogger
from .train_classifier import train_encoder


__all__ = [
    "run",
]

def _collect_prototypes(backbone: nn.Module, loader, n_classes: int, device) -> torch.Tensor:
    """Compute L2‑normalised class means from *non‑augmented* loader."""
    sums, cnts = None, torch.zeros(n_classes, device=device)
    with torch.no_grad():
        for imgs, lbls in loader:
            feats = backbone(imgs.to(device))  # (B,D)
            if sums is None:
                sums = torch.zeros(n_classes, feats.size(1), device=device)
            for f, y in zip(feats, lbls.to(device)):
                sums[y] += f;  cnts[y] += 1
    protos = F.normalize(sums / cnts.unsqueeze(1), dim=1)
    return protos  # (C,D)

def _make_cutmix_criterion(use_cutmix: bool = True) -> callable:
    ce = nn.CrossEntropyLoss()

    def criterion(pred, targets):
        if use_cutmix and isinstance(targets, tuple):
            t1, t2, lam = targets
            return lam * ce(pred, t1) + (1.0 - lam) * ce(pred, t2)
        return ce(pred, targets)

    return criterion


def run(
    backbone: nn.Module,
    classifier: nn.Module,
    loaders: Dict[str, object],
    epochs: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    logger: CSVLogger | None = None,
    *,
    use_cutmix: bool = True,
) -> Tuple[nn.Module, nn.Module, torch.Tensor]:
    """Train ``backbone`` + ``classifier`` jointly.

    Parameters
    ----------
    backbone : nn.Module
        Encoder network (e.g. ResNet‑18).
    classifier : nn.Module
        CosineClassifier (or any nn.Module mapping features -> logits).
    loaders : dict[str, DataLoader]
        Expect keys: ``d0_train``, ``d0_test``, ``proto``.
    epochs : int
        Number of training epochs.
    device : torch.device
        CUDA / CPU device.
    optimizer, scheduler : torch.optim.*
        Optimiser and LR scheduler initialised on *both* modules' params.
    logger : CSVLogger | None
        Optional CSV logger.
    use_cutmix : bool, default True
        Whether the loaders emit CutMix tuples.

    Returns
    -------
    backbone, classifier, prototypes
        The two trained sub‑modules (with gradients) and the L2‑normalised
        class prototype tensor with shape (C, D).
    """

    # Compose a single model for the generic train loop
    model = nn.Sequential(backbone, classifier).to(device)

    cutmix_criterion = _make_cutmix_criterion(use_cutmix=use_cutmix)
    ce = nn.CrossEntropyLoss()

    # ---- Training ---------------------------------------------------------
    train_encoder(
        model,
        cutmix_criterion,
        loaders,
        ce,
        optimizer,
        scheduler,
        device,
        num_epochs=epochs,
        logger=logger,
    )

    # ---- Prototypes -------------------------------------------------------
    backbone.eval()  # inference mode for prototype extraction
    with torch.no_grad():
        prototypes = _collect_prototypes(
            backbone, loaders["proto"], C.BASE_CLASS, device
        )

    return backbone, classifier, prototypes
