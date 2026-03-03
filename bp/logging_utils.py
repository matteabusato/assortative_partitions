from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional
import json
import os

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def wandb_init_if_enabled(
    *,
    use_wandb: bool,
    project: str,
    name: str,
    group: Optional[str],
    config: Dict[str, Any],
) -> None:
    """
    Initializes W&B if enabled and available. Otherwise does nothing.
    """
    if not use_wandb:
        return
    if not WANDB_AVAILABLE:
        raise ImportError("wandb is not installed in this environment. `pip install wandb` first.")

    wandb.init(
        project=project,
        name=name,
        group=group,
        config=config,
    )


def wandb_finish_if_enabled(use_wandb: bool) -> None:
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()


def wandb_set_summary_if_enabled(use_wandb: bool, summary: Dict[str, Any]) -> None:
    if use_wandb and WANDB_AVAILABLE:
        wandb.summary.update(summary)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _default(o):
        try:
            import numpy as np
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
        except Exception:
            pass
        return str(o)

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_default)


def wandb_save_if_enabled(use_wandb: bool, path: str) -> None:
    if use_wandb and WANDB_AVAILABLE:
        wandb.save(path)