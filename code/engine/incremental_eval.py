
#!/usr/bin/env python3
"""
Runs FSCIL incremental sessions using trained BiAG
• loads checkpoints produced by base_train.py & train_biag.py
• prints per-session accuracy and forgetting metrics
"""
import torch, torch.nn as nn
from collections import defaultdict
import code.config as C
from code.model.backbone import ResNet12, ResNet18
from code.model.classifier import CosineClassifier
from code.utils.session_state import SessionState
from code.model.BiAG import BiAGWrapper
from pathlib import Path


@torch.no_grad()
def _classification_metrics(model, loader, device, seen_ids, base_ids):
    model.eval()
    correct = total = 0
    base_correct = base_total = novel_correct = novel_total = 0
    seen = torch.tensor(sorted(seen_ids), device=device, dtype=torch.long)
    base = torch.tensor(sorted(base_ids), dtype=torch.long)
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device)) # (B, C) raw scores
            # Restrict the classifier to classes introduced so far. Map column
            # indices back to global class IDs (also works for noncontiguous IDs).
            preds = seen[logits[:, seen].argmax(1)].cpu()
            hits = preds == y
            is_base = torch.isin(y, base)
            correct += hits.sum().item()
            total += y.size(0)
            base_correct += hits[is_base].sum().item()
            base_total += is_base.sum().item()
            novel_correct += hits[~is_base].sum().item()
            novel_total += (~is_base).sum().item()
    return {
        "acc": 100 * correct / total if total else 0.0,
        "base_acc": 100 * base_correct / base_total if base_total else None,
        "novel_acc": 100 * novel_correct / novel_total if novel_total else None,
    }

def load_state(args=None):
    device = torch.device(C.DEVICE if torch.cuda.is_available() else "cpu")

    exp_dir = Path(args.output_dir) / args.dataset
    exp_dir.mkdir(parents=True, exist_ok=True)

    # backbone
    backbone_path = Path(args.pt_backbone) if args.pt_backbone else exp_dir / "backbone_pt_last.pt"
    backbone = ResNet18() if C.BACKBONE_MODEL.lower()=="resnet18" else ResNet12()
    backbone.load_state_dict(torch.load(backbone_path, map_location=device))
    backbone = backbone.to(device).eval()

    # classifier
    TOTAL_C = C.NUM_CLASSES
    clf = CosineClassifier(in_dim=backbone.out_dim,
                           num_classes=TOTAL_C,
                           init_method="l2",
                           learnable_scale=True).to(device)
    clf.weight.data.zero_()
    clf_path = Path(args.pt_classifier) if args.pt_classifier else exp_dir / "classifier_pt_last.pt"
    clf_sd = torch.load(clf_path, map_location=device, weights_only=True)
    base_w = clf_sd["weight"]
    if base_w.shape != (C.BASE_CLASS, backbone.out_dim):
        raise ValueError("Classifier checkpoint must contain exactly the base classes")
    clf.weight.data[:base_w.size(0)] = base_w      # base 60×D
    clf.scale.data.copy_(clf_sd["scale"])

    # prototypes
    proto_path = Path(args.pt_proto) if args.pt_proto else exp_dir / "proto_pt_last.pt"
    protos = torch.load(proto_path, map_location=device).to(device)
    if protos.shape != base_w.shape:
        raise ValueError("Base prototypes must match the base classifier weights")
    if protos.size(0) < TOTAL_C:                   # pad → 100×D
        pad = torch.zeros(TOTAL_C - protos.size(0),
                          backbone.out_dim, device=device)
        protos = torch.cat([protos, pad], dim=0)

    print(f"proto_path :{proto_path}")
    # BiAG
    biag = BiAGWrapper(backbone.out_dim).to(device)
    biag_path = Path(args.biag) if args.biag else exp_dir / "biag_pt_last.pt"
    print(f"biag_path :{biag_path}")
    biag_sd = torch.load(biag_path, map_location=device, weights_only=True)
    if "biag.scm.mlp.0.weight" not in biag_sd:
        raise ValueError("Legacy BiAG checkpoint: retrain BiAG with the corrected episode/SCM implementation. Base checkpoints can be reused.")
    biag.load_state_dict(biag_sd)
    biag.eval()

    # session state
    state = SessionState(backbone, clf)
    state.biag   = biag
    state.protos = protos
    state.weights = [clf.weight.data]   # for compatibility
    state.device  = device
    return state

@torch.no_grad()
def evaluate(loaders, state):
    if not (len(loaders["support_sess"]) == len(loaders["test_sess"])
            == len(loaders["class_splits"]["sessions"])):
        raise ValueError("Support, test and class-split session counts must match")
    device   = state.device
    backbone = state.backbone
    clf      = state.classifier
    joint    = nn.Sequential(backbone.eval(), clf.eval())

    acc_sess, forgetting = [], []
    base_acc_sess, novel_acc_sess = [], []
    base_ids = set(loaders["class_splits"]["base"])
    prev_seen = set(base_ids)

    # base
    metrics = _classification_metrics(joint, loaders["d0_test"], device, prev_seen, base_ids)
    acc0 = metrics["acc"]
    print(f"[sess0] acc={acc0:5.2f}")
    acc_sess.append(acc0)
    base_acc_sess.append(metrics["base_acc"])
    novel_acc_sess.append(metrics["novel_acc"])

    # incremental
    for s,(sup_loader,test_loader) in enumerate(
            zip(loaders["support_sess"], loaders["test_sess"]),1):

        cumul_ids  = loaders["class_splits"]["sessions"][s-1]
        new_ids    = [gid for gid in cumul_ids if gid not in prev_seen]
        old_ids = sorted(prev_seen)
        if not new_ids:
            raise ValueError(f"Session {s} introduces no new classes")

        imgs_by_cls = defaultdict(list)
        for x_b, y_b in sup_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            for x, y in zip(x_b, y_b):
                imgs_by_cls[int(y)].append(x)

        if set(imgs_by_cls) != set(new_ids):
            raise ValueError(f"Session {s}: support labels must match new class IDs")
        if any(len(imgs_by_cls[gid]) != C.SHOT for gid in new_ids):
            raise ValueError(f"Session {s}: expected {C.SHOT} support images per class")
        p_new = torch.stack([state.extract_proto(torch.stack(imgs_by_cls[gid]))
                             for gid in new_ids], dim=1)            # (1,N,D)

        p_old = state.protos[old_ids].unsqueeze(0)
        w_old = clf.weight.data[old_ids].unsqueeze(0)
        new_w = state.biag(p_new, p_old, w_old)                  # (N,D), including N=1
        if new_w.shape != p_new.shape[1:]:
            raise ValueError("Generated weights must have shape (new_classes, features)")

        for i,gid in enumerate(new_ids):
            clf.weight.data[gid] = new_w[i]
            state.protos[gid]    = p_new[0,i]
        state.weights[0] = clf.weight.data
        prev_seen.update(new_ids)

        metrics = _classification_metrics(joint, test_loader, device, prev_seen, base_ids)
        acc = metrics["acc"]
        acc_sess.append(acc)
        base_acc_sess.append(metrics["base_acc"])
        novel_acc_sess.append(metrics["novel_acc"])
        # Retention must compare the same base population across sessions.
        forgetting.append(acc0 - metrics["base_acc"])
        print(f"[sess{s}] acc={acc:5.2f}  forget={forgetting[-1]:5.2f}")

    mean_all = sum(acc_sess)/len(acc_sess)
    mean_inc = sum(acc_sess[1:])/len(acc_sess[1:]) if len(acc_sess)>1 else 0.0
    return {"acc_sessions":acc_sess,
            "base_acc_sessions":base_acc_sess,
            "novel_acc_sessions":novel_acc_sess,
            "forgetting":forgetting,
            "mean_all":mean_all,
            "mean_inc":mean_inc}
