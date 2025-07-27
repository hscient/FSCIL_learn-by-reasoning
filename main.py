#!/usr/bin/env python3
"""
Unified CLI for FSCIL + BiAG pipeline.

Usage:
  python scripts/main.py base               --dataset cifar100 ...
  python scripts/main.py biag               --dataset cifar100 ...
  python scripts/main.py incremental_run    --dataset cifar100 ...
  python scripts/main.py all                --dataset cifar100 ...
"""
import argparse
import importlib
from scripts.common_argparser import build_parser as common_build_parser


STAGES = ("base", "biag", "incremental_run", "all")


def build_main_parser() -> argparse.ArgumentParser:
    # 공통 인자 파서(부모) 재사용
    common = common_build_parser()

    p = argparse.ArgumentParser(
        prog="scripts/main.py",
        description="Unified CLI for FSCIL + BiAG"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_stage(name: str):
        return sub.add_parser(name, parents=[common], add_help=False)

    add_stage("base")
    add_stage("biag")
    add_stage("incremental_run")
    add_stage("all")

    for s in ("biag", "all"):
        g = sub.choices[s]
        g.add_argument("--biag_epochs", type=int, help="override C.BIAG_EPOCHS")
        g.add_argument("--biag_lr",     type=float, help="override C.BIAG_LR")
        g.add_argument("--biag_depth",  type=int, help="override depth if used in your impl")

    return p


def import_and_run(module_name: str, ns: argparse.Namespace):
    """scripts.<module_name>.main(ns)를 호출"""
    mod = importlib.import_module(f"scripts.{module_name}")
    return mod.main(ns)


def main():
    parser = build_main_parser()
    args = parser.parse_args()

    cmd = args.cmd

    pass_args = {k: v for k, v in vars(args).items() if k != "cmd"}
    ns = argparse.Namespace(**pass_args)

    if cmd == "base":
        import_and_run("base", ns)
    elif cmd == "biag":
        import_and_run("biag", ns)
    elif cmd == "incremental_run":
        import_and_run("incremental_run", ns)
    elif cmd == "all":
        import_and_run("base", ns)
        import_and_run("biag", ns)
        import_and_run("incremental_run", ns)
    else:
        parser.error(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
