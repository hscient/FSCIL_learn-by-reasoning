#!/usr/bin/env python3
"""Incremental session evaluation ("run" stage).
Loads base‑session artefacts + trained BiAG, then runs
all FSCIL sessions and stores per‑session metrics in CSV.
"""
from __future__ import annotations
import torch, argparse, json
from pathlib import Path
from code.utils.logger import CSVLogger
from code.data.fscil_loader_legacy import build_cifar_fscil_loaders, build_mini_imagenet_loaders
from code.engine import incremental_eval as IE
import code.config as C
from scripts.common_argparser import build_parser


def main(cli_args: argparse.Namespace | None = None):
    if cli_args is None:
        cli_args = build_parser().parse_args()
    C.update_from_args(vars(cli_args))
    if getattr(cli_args, "show_config", False):
        C.print_effective_config()

    exp_run = Path(cli_args.output_dir) / cli_args.dataset / "run"
    exp_run.mkdir(parents=True, exist_ok=True)
    metrics_csv = exp_run / "metrics.csv"

    if C.DATASET_NAME == "cifar100":
        loaders = build_cifar_fscil_loaders(seed=C.SEED)
    elif C.DATASET_NAME == "miniimagenet":
        loaders = build_mini_imagenet_loaders(seed=C.SEED)

    # session state (backbone+biag+tables)
    state = IE.load_state(cli_args)

    results = IE.evaluate(loaders, state)

    # write log
    logger = CSVLogger(metrics_csv, fieldnames=["stage","session","acc","forget","mean_all","mean_inc"])
    for i, acc in enumerate(results["acc_sessions"]):
        fg = 0.0 if i==0 else results["forgetting"][i-1]
        logger.log(stage="run", session=i, acc=acc, forget=fg)
    # aggregate rows
    logger.log(stage="run", session="mean_all", acc=results["mean_all"], forget="-")
    logger.log(stage="run", session="mean_inc", acc=results["mean_inc"], forget="-")

    with open(exp_run/"summary.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"[run] finished ✓  results saved to {metrics_csv}")


if __name__ == "__main__":
    main()
