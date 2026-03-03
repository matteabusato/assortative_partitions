from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import numpy as np


InitMode = Literal["uniform", "unif_diag", "one_hot", "one_hot_softmax", "personalized", "gaussian"]


@dataclass(frozen=True)
class ChiInitSpec:
    mode: InitMode = "unif_diag"
    epsilon: float = 0.01
    softmax_eps: float = 1e-2   # used for one_hot_softmax


def make_chi_init(spec: ChiInitSpec, seed: Optional[int] = None) -> np.ndarray:
    """
    Returns chi_init with shape (3,3), normalized to sum to 1.

    Notes:
    - Uses numpy RNG seed if provided.
    - Keeps your original semantics for each initialization mode.
    """
    if seed is not None:
        np.random.seed(seed)

    if spec.mode == "uniform":
        chi = np.ones((3, 3), dtype=float)

    elif spec.mode == "unif_diag":
        chi = np.zeros((3, 3), dtype=float)
        chi[0, 0] = 1 / 3
        chi[1, 1] = 1 / 3
        chi[2, 2] = 1 / 3

    elif spec.mode == "one_hot":
        chi = np.zeros((3, 3), dtype=float)
        chi[0, 0] = 1.0

    elif spec.mode == "one_hot_softmax":
        e = float(spec.softmax_eps)
        chi = np.full((3, 3), e, dtype=float)
        chi[0, 0] = 1.0 - 8 * e  # same as your code

    elif spec.mode == "personalized":
        eps = float(spec.epsilon)
        chi = np.zeros((3, 3), dtype=float)
        chi[0, 0] = 1.0 - eps
        chi[1, 1] = eps / 2
        chi[2, 2] = eps / 2

    elif spec.mode == "gaussian":
        chi = np.random.rand(3, 3)

    else:
        raise ValueError(f"Unknown chi init mode: {spec.mode}")

    s = float(np.sum(chi))
    if s <= 0:
        raise ValueError("chi initialization has non-positive sum; cannot normalize.")
    chi /= s
    return chi