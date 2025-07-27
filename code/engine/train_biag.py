"""BiAG weight‑generator training loop.
Returns (best_state_dict, final_state_dict, best_cos)"""
from __future__ import annotations
import torch, torch.nn.functional as F
from typing import Tuple
from code.utils.logger import CSVLogger
from code.utils.session_state import SessionState
from code.model.BiAG import BiAG  # low‑level core
import random, math


def _random_episode(state: SessionState, n_base: int, k_way: int = 5):
    all_cls = list(range(n_base))
    new_ids = sorted(random.sample(all_cls, k_way))
    old_ids = [c for c in all_cls if c not in new_ids]

    p_new = state.protos[new_ids].unsqueeze(1)            # (K ,1 ,D)
    gt_w  = state.weights[0][new_ids]
    p_old = state.protos[old_ids].unsqueeze(0).expand(p_new.size(0), -1, -1)
    w_old = state.weights[0][old_ids].unsqueeze(0).expand_as(p_old)

    p_new = F.normalize(p_new, dim=-1)
    p_old = F.normalize(p_old, dim=-1)
    w_old = F.normalize(w_old, dim=-1)
    gt_w  = F.normalize(gt_w, dim=-1)
    return p_new, gt_w, p_old, w_old

def run(
    state: SessionState,
    biag_wrapper: torch.nn.Module,
    epochs: int,
    episodes_per_epoch: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    logger: CSVLogger | None = None,
) -> Tuple[dict, dict, float]:
    """Train BiAG. Returns best & final state_dict and best cosine."""
    device = state.device
    biag_wrapper.train()
    best_cos = -1.0
    best_state = None

    n_base = state.protos.size(0)
    for ep in range(1, epochs + 1):
        epoch_losses = []
        epoch_coss   = []
        for _ in range(episodes_per_epoch):
            p_new, gt_w, p_old, w_old = _random_episode(state, n_base)
            pred_w = biag_wrapper(p_new.to(device), p_old.to(device), w_old.to(device))
            cos = F.cosine_similarity(pred_w, gt_w.to(device), dim=-1).mean()
            loss = 1 - cos
            optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
            epoch_losses.append(loss.item()); epoch_coss.append(cos.item())

        mean_loss = sum(epoch_losses)/len(epoch_losses)
        mean_cos  = sum(epoch_coss)/len(epoch_coss)
        if logger:
            logger.log(stage="biag", epoch=ep, loss=mean_loss, cos=mean_cos)
        if mean_cos > best_cos:
            best_cos = mean_cos
            best_state = {k: v.clone().cpu() for k, v in biag_wrapper.state_dict().items()}
        print(f"[biag] epoch {ep:02d}/{epochs}  loss={mean_loss:.4f}  cos={mean_cos:.4f}")

    final_state = {k: v.clone().cpu() for k, v in biag_wrapper.state_dict().items()}
    return best_state, final_state, best_cos
