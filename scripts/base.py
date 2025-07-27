"""CLI entry point for joint backbone + classifier training.

Usage example (from project root)::

    python3 -m scripts.joint \
        --dataset CIFAR100 --output_dir ./experiments/cifar
"""
import argparse
from pathlib import Path
import torch

from code.utils.utils import confirm_overwrite
from code import config as C
from code.data.base_loader import build_fscil_loaders
from code.model.backbone import ResNet12, ResNet18
from code.model.classifier import CosineClassifier
from code.utils.logger import CSVLogger
from code.engine import base_train as E
from scripts.common_argparser import build_parser  # re‑use global parser


def main(args: argparse.Namespace | None = None):
    if args is None:
        args = build_parser().parse_args()

    C.update_from_args(vars(args))
    if getattr(args, "show_config", False):
        C.print_effective_config()
        if getattr(args, "save_config", None):
            C.save_effective_config(args.save_config)

    device = torch.device(C.DEVICE if torch.cuda.is_available() else "cpu")

    exp_dir = Path(args.output_dir) / args.dataset
    exp_dir.mkdir(parents=True, exist_ok=True)

    pt_backbone = Path(args.pt_backbone) if args.pt_backbone else exp_dir / "backbone_pt_last.pt"
    pt_classifier = Path(args.pt_classifier) if args.pt_classifier else exp_dir / "classifier_pt_last.pt"
    pt_proto = Path(args.pt_proto) if args.pt_proto else exp_dir / "proto_pt_last.pt"
    print(f"pt_backbone:{pt_backbone}, pt_classifier:{pt_classifier}, pt_proto:{pt_proto}")

    for p in (pt_backbone, pt_classifier, pt_proto):
        p.parent.mkdir(parents=True, exist_ok=True)

    if confirm_overwrite([pt_backbone, pt_classifier, pt_proto], tag="biag"):
        print("[biag] Skipping.")
        return

    # Data & model
    loaders = build_fscil_loaders(dataset_name=args.dataset, seed=C.SEED)
    n_base = len(loaders["class_splits"]["base"])

    backbone = ResNet18() if C.BACKBONE_MODEL.lower() == "resnet18" else ResNet12()
    classifier = CosineClassifier(
        in_dim=backbone.out_dim,
        num_classes=n_base,
        init_method="l2",
        learnable_scale=True,
    )

    params = list(backbone.parameters()) + list(classifier.parameters())
    opt = torch.optim.SGD(params, lr=C.INIT_LR, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.EPOCHS)

    logger = CSVLogger(exp_dir / "log.csv", fieldnames=["stage", "epoch", "acc", "loss", "lr"])

    # Train
    backbone.to(device)
    classifier.to(device)

    backbone, classifier, protos = E.run(
        backbone,
        classifier,
        loaders,
        C.EPOCHS,
        device,
        opt,
        sch,
        logger,
        use_cutmix=True,
    )

    torch.save(backbone.state_dict(), pt_backbone)
    torch.save(classifier.state_dict(), pt_classifier)
    torch.save(protos.cpu(), pt_proto)

    print("[joint] training complete ✓ — checkpoints saved to", exp_dir)


if __name__ == "__main__":  # pragma: no cover
    main()