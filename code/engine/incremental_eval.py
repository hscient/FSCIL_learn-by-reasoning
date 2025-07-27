
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
def _accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device)) # (B, C) raw scores
            preds = logits.argmax(1).cpu()
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 0 if total == 0 else correct / total * 100

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
    base_w = torch.load(clf_path, map_location=device)["weight"]
    clf.weight.data[:base_w.size(0)] = base_w      # base 60×D

    # prototypes
    proto_path = Path(args.pt_proto) if args.pt_proto else exp_dir / "proto_pt_last.pt"
    protos = torch.load(proto_path, map_location=device).to(device)
    if protos.size(0) < TOTAL_C:                   # pad → 100×D
        pad = torch.zeros(TOTAL_C - protos.size(0),
                          backbone.out_dim, device=device)
        protos = torch.cat([protos, pad], dim=0)

    # BiAG
    biag = BiAGWrapper(backbone.out_dim).to(device)
    biag_path = Path(args.biag) if args.biag else exp_dir / "biag_pt_last.pt"
    print(f"biag_path :{biag_path}")
    biag.load_state_dict(torch.load(biag_path, map_location=device))
    biag.eval()

    # session state
    state = SessionState(backbone, clf)
    state.biag   = biag
    state.protos = protos
    state.weights = [clf.weight.data]   # for compatibility
    state.device  = device
    return state

def evaluate(loaders, state):
    device   = state.device
    backbone = state.backbone
    clf      = state.classifier
    joint    = nn.Sequential(backbone.eval(), clf.eval())

    acc_sess, forgetting = [], []

    # base
    acc0 = _accuracy(joint, loaders["d0_test"], device)
    acc_sess.append(acc0)
    prev_seen = set(loaders["class_splits"]["base"])

    # incremental
    for s,(sup_loader,test_loader) in enumerate(
            zip(loaders["support_sess"], loaders["test_sess"]),1):

        cumul_ids  = loaders["class_splits"]["sessions"][s-1]
        new_ids    = [gid for gid in cumul_ids if gid not in prev_seen]
        prev_seen.update(new_ids)

        imgs_by_cls = defaultdict(list)
        for x_b, y_b in sup_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            for x, y in zip(x_b, y_b):
                imgs_by_cls[int(y)].append(x)

        p_new = torch.stack([state.extract_proto(torch.stack(imgs_by_cls[gid]))
                             for gid in new_ids], dim=1)            # (1,N,D)

        p_old = state.protos.unsqueeze(0)
        w_old = clf.weight.data.unsqueeze(0)
        new_w = state.biag(p_new, p_old, w_old).squeeze(0)          # (N,D)

        for i,gid in enumerate(new_ids):
            clf.weight.data[gid] = new_w[i]
            state.protos[gid]    = p_new[0,i]
        state.weights[0] = clf.weight.data

        acc = _accuracy(joint, test_loader, device)
        acc_sess.append(acc)
        forgetting.append(acc0 - acc)
        print(f"[sess{s}] acc={acc:5.2f}  forget={forgetting[-1]:5.2f}")

    mean_all = sum(acc_sess)/len(acc_sess)
    mean_inc = sum(acc_sess[1:])/len(acc_sess[1:]) if len(acc_sess)>1 else 0.0
    return {"acc_sessions":acc_sess,
            "forgetting":forgetting,
            "mean_all":mean_all,
            "mean_inc":mean_inc}
