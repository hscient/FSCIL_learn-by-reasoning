#!/usr/bin/env python3
import torch, argparse
from pathlib import Path
from code.utils.utils import confirm_overwrite
from code import config as C
from code.utils.logger import CSVLogger
from code.utils.session_state import SessionState
from code.model.backbone import ResNet12, ResNet18
from code.model.classifier import CosineClassifier
from code.engine import train_biag as E
from code.model.BiAG import BiAGWrapper  # reuse wrapper class
from scripts.common_argparser import build_parser


def main(args=None):
    if args is None:
        args = build_parser().parse_args()

    C.update_from_args(vars(args))
    if getattr(args, "show_config", False):
        C.print_effective_config()
        if getattr(args, "save_config", None):
            C.save_effective_config(args.save_config)

    device = torch.device(C.DEVICE if torch.cuda.is_available() else "cpu")
    exp_dir = Path(args.output_dir) / args.dataset; exp_dir.mkdir(parents=True, exist_ok=True)

    pt_best = Path(args.pt_biag_best) if args.pt_biag_best else exp_dir / "biag_pt_best.pt"
    pt_last = Path(args.pt_biag_last) if args.pt_biag_last else exp_dir / "biag_pt_last.pt"
    for p in (pt_best, pt_last):
        p.parent.mkdir(parents=True, exist_ok=True)

    if confirm_overwrite([pt_best, pt_last], tag="biag"):
        print("[biag] Skipping.")
        return

    backbone = ResNet18() if C.BACKBONE_MODEL.lower()=="resnet18" else ResNet12()
    backbone_path = Path(args.pt_backbone) if args.pt_backbone else Path(args.output_dir)/args.dataset/"backbone_pt_last.pt"
    backbone.load_state_dict(torch.load(backbone_path, map_location=device))
    backbone = backbone.to(device).eval()

    clf_path = Path(args.pt_classifier) if args.pt_classifier else Path(args.output_dir)/args.dataset/"classifier_pt_last.pt"
    clf_sd = torch.load(clf_path, map_location=device)
    classifier = CosineClassifier(in_dim=backbone.out_dim, num_classes=clf_sd["weight"].shape[0], learnable_scale=True).to(device)
    classifier.load_state_dict(clf_sd)

    state = SessionState(backbone, classifier)
    proto_path = Path(args.pt_proto) if args.pt_proto else Path(args.output_dir)/args.dataset/"proto_pt_last.pt"
    state.protos = torch.load(proto_path, map_location=device).to(device)

    # BiAG wrapper & optimiser
    biag = BiAGWrapper(backbone.out_dim).to(device)
    base_lr = C.BIAG_LR; wd=1e-2
    # dec_emged, alpha is optional (default : False)
    high_lr_keywords = {"dec_embed","alpha"}; no_wd_keywords={"gamma"}
    decay_params, no_decay_params, dec_embed_params = [], [], []
    for n,p in biag.named_parameters():
        if any(k in n for k in high_lr_keywords): dec_embed_params.append(p)
        elif any(k in n for k in no_wd_keywords): no_decay_params.append(p)
        else: decay_params.append(p)
    optim = torch.optim.AdamW([
        {"params":decay_params,"lr":base_lr,"weight_decay":wd},
        {"params":no_decay_params,"lr":base_lr,"weight_decay":0.0},
        {"params":dec_embed_params,"lr":base_lr*10,"weight_decay":0.0}], betas=(0.9,0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=C.BIAG_EPOCHS*C.PSEUDO_EPISODES_PER_EPOCH)

    logger = CSVLogger(exp_dir/"log.csv", fieldnames=["stage","epoch","loss","cos"])

    best_state, final_state, best_cos = E.run(state, biag, C.BIAG_EPOCHS, C.PSEUDO_EPISODES_PER_EPOCH, optim, sched, logger)
    torch.save(best_state, pt_best); torch.save(final_state, pt_last)
    print(f"[biag] finished ✓  best cos={best_cos:.4f}")

if __name__ == "__main__":
    main()
