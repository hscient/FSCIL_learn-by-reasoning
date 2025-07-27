import argparse
import sys


# def in_colab() -> bool:
#     return "google.colab" in sys.modules

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("FSCIL – base training")

    p.add_argument("--dataset", required=True, choices=["cifar100", "miniimagenet"],
                   help="Dataset name (case‑insensitive)")
    p.add_argument("--data_root")
    # p.add_argument("--data_root", default="./data")
    p.add_argument("--output_dir", default="./checkpoints")
    p.add_argument("--seed", type=int, default=1)

    # Hyper‑params (override config.py)
    p.add_argument("--epochs", type=int)
    p.add_argument("--init_lr", type=float)
    p.add_argument("--backbone_model", type=str, choices=["resnet12", "resnet18"])
    p.add_argument("--batch_size", type=int)
    p.add_argument("--num_workers", type=int)
    p.add_argument("--use_cutmix", action="store_true", default=False)

    # base pt (backbone, classifier, proto) path
    p.add_argument(
        "--pt_backbone",
        type=str,
        default=None,
        help="saved backbone state_dict path",
    )
    p.add_argument(
        "--pt_classifier",
        type=str,
        default=None,
        help="saved classifier state_dict path",
    )
    p.add_argument(
        "--pt_proto",
        type=str,
        default=None,
        help="saved base‑session prototypes path",
    )

    # biag
    p.add_argument(
        "--biag",
        type=str,
        default=None,
        help="BiAG (.pt) file path. if given, it overwrite existing file.",
    )

    p.add_argument(
        "--pt_biag_best",
        type=str,
        default=None,
        help="best cos BiAG state_dict save path",
    )
    p.add_argument(
        "--pt_biag_last",
        type=str,
        default=None,
        help="final epoch BiAG state_dict save path",
    )

    # show & save config
    p.add_argument(
        "--show_config",action=argparse.BooleanOptionalAction, default=False,
        help=f"Print effective config (default: True)")
    p.add_argument("--save_config", type=str, default=None,
                   help="config value save path")

    return p