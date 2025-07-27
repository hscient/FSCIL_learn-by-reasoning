from __future__ import annotations

"""Utility helpers for safely handling existing checkpoint files.

Usage::

    from pathlib import Path
    from overwrite_utils import confirm_overwrite

    exp_dir = Path(args.output_dir) / args.dataset
    exp_dir.mkdir(parents=True, exist_ok=True)

    pt_best = exp_dir / "biag_pt_best.pt"
    pt_last = exp_dir / "biag_pt_last.pt"

    # Ask the user whether to overwrite existing checkpoints. If the function
    # returns True we should *skip* further work; otherwise continue.
    if confirm_overwrite([pt_best, pt_last], tag="biag"):
        return

The helper can be reused anywhere you need the same "exist → skip/overwrite" flow.
"""

from pathlib import Path
from typing import Sequence

__all__ = [
    "confirm_overwrite",
]

def _all_exist(paths: Sequence[Path]) -> bool:  # pragma: no cover
    """Return ``True`` iff *all* paths exist on disk."""
    return all(p.exists() for p in paths)


def _any_exist(paths: Sequence[Path]) -> bool:  # pragma: no cover
    """Return ``True`` iff *any* of the paths exist on disk."""
    return any(p.exists() for p in paths)


def confirm_overwrite(
    paths: Sequence[Path],
    *,
    tag: str | None = None,
    require_all: bool = True,
    default_skip: bool = True,
) -> bool:
    """Prompt the user whether to overwrite existing files.

    Parameters
    ----------
    paths
        A list / tuple of :class:`~pathlib.Path` objects to inspect.
    tag
        Short label printed in prompt messages (e.g. *"biag"*). ``None`` for no label.
    require_all
        When *True* (default), the prompt is shown only if **all** paths exist.
        Set to *False* to prompt when *any* of the paths already exist.
    default_skip
        Determines the default behaviour when running in a *non‑interactive* context
        (no TTY) **or** when the user simply hits *ENTER* without typing a response.
        *True* keeps the original "skip" behaviour; *False* overwrites.

    Returns
    -------
    bool
        *True*  → caller **should skip** its work (user declined to overwrite).
        *False* → caller may proceed and safely overwrite / recreate files.
    """
    exists_fn = _all_exist if require_all else _any_exist

    if not exists_fn(paths):
        # Nothing to overwrite; we can safely continue.
        return False

    names = ", ".join(p.name for p in paths)
    label = f"[{tag}] " if tag else ""

    # Construct prompt string with sensible default indication.
    default_letter = "N" if default_skip else "Y"
    opposite_letter = "y" if default_skip else "n"
    prompt = f"{label}{names} exist. Overwrite? [{default_letter}/{opposite_letter}]: "

    try:
        answer = input(prompt)
    except EOFError:
        # Running in a non‑interactive environment; fall back to default.
        return default_skip

    answer = answer.strip().lower()
    if not answer:
        # Empty input → default choice
        return default_skip

    # Return True (skip) unless the user explicitly said "y".
    return not answer.startswith("y")
