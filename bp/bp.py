from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import math
import time
from scipy.optimize import least_squares

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class BPConfig:
    N: int
    D: int
    H: int
    M_target: np.ndarray               # shape (3,)
    threshold: float = 1e-21
    max_iter: int = 10_000_000
    log_every: int = 1000
    damping: float = 0.01

    # mu solver options
    settingmu: str = "previous"        # "always_zero", "zero", "previous"
    loss_mu: str = "soft_l1"           # "linear", "soft_l1", "huber", "cauchy", "arctan"

    # logging
    use_wandb: bool = False


class BeliefPropagation:
    """
    Belief Propagation fixed-point iteration for K=3 groups.

    State:
      - chi: (3,3) message matrix
      - mu:  (3,) Lagrange multipliers (solved by least_squares)
    """

    K: int = 3

    def __init__(
        self,
        cfg: BPConfig,
        chi_init: np.ndarray,
        mu_init: Optional[np.ndarray] = None,
    ):
        self.cfg = cfg

        chi_init = np.array(chi_init, dtype=float, copy=True)
        if chi_init.shape != (self.K, self.K):
            raise ValueError(f"chi_init must have shape {(self.K, self.K)}, got {chi_init.shape}")
        self.chi = self._normalize_chi(chi_init)

        if mu_init is None:
            mu_init = np.zeros(self.K, dtype=float)
        self.mu = np.array(mu_init, dtype=float, copy=True)
        if self.mu.shape != (self.K,):
            raise ValueError(f"mu_init must have shape {(self.K,)}, got {self.mu.shape}")

    @staticmethod
    def _assign_f(i: int) -> Tuple[int, int, int]:
        if i == 0:
            return 0, 1, 2
        elif i == 1:
            return 1, 2, 0
        else:
            return 2, 0, 1

    @staticmethod
    def _normalize_chi(chi: np.ndarray) -> np.ndarray:
        s = float(np.sum(chi))
        if s > 1e-300:
            chi = chi / s
        return chi

    def _find_current_mu(
        self,
        chi: np.ndarray,
        mu0: np.ndarray,
    ) -> np.ndarray:
        D = self.cfg.D
        M_star = self.cfg.M_target
        loss = self.cfg.loss_mu
        settingmu = self.cfg.settingmu

        def m_values(mu: np.ndarray) -> np.ndarray:
            mu1, mu2, mu3 = mu

            den = (
                np.exp((2 / D) * mu1) * (chi[0, 0] ** 2)
                + np.exp((2 / D) * mu2) * (chi[1, 1] ** 2)
                + np.exp((2 / D) * mu3) * (chi[2, 2] ** 2)
                + 2
                * (
                    np.exp((1 / D) * (mu1 + mu2)) * chi[0, 1] * chi[1, 0]
                    + np.exp((1 / D) * (mu2 + mu3)) * chi[1, 2] * chi[2, 1]
                    + np.exp((1 / D) * (mu3 + mu1)) * chi[2, 0] * chi[0, 2]
                )
            )

            return np.array(
                [
                    (
                        np.exp((2 / D) * mu1) * (chi[0, 0] ** 2)
                        + np.exp((1 / D) * (mu1 + mu2)) * (chi[0, 1] * chi[1, 0])
                        + np.exp((1 / D) * (mu1 + mu3)) * (chi[0, 2] * chi[2, 0])
                    )
                    / den,
                    (
                        np.exp((2 / D) * mu2) * (chi[1, 1] ** 2)
                        + np.exp((1 / D) * (mu2 + mu1)) * (chi[1, 0] * chi[0, 1])
                        + np.exp((1 / D) * (mu2 + mu3)) * (chi[1, 2] * chi[2, 1])
                    )
                    / den,
                    (
                        np.exp((2 / D) * mu3) * (chi[2, 2] ** 2)
                        + np.exp((1 / D) * (mu3 + mu1)) * (chi[2, 0] * chi[0, 2])
                        + np.exp((1 / D) * (mu3 + mu2)) * (chi[2, 1] * chi[1, 2])
                    )
                    / den,
                ],
                dtype=float,
            )

        def residuals(mu: np.ndarray) -> np.ndarray:
            return m_values(mu) - M_star

        if settingmu == "zero":
            mu0 = np.zeros(3, dtype=float)
        elif settingmu == "previous":
            mu0 = mu0
        elif settingmu == "always_zero":
            return np.zeros(3, dtype=float)

        res = least_squares(
            residuals,
            mu0,
            method="trf",
            loss=loss,
            xtol=2.23e-16,
            ftol=2.23e-16,
            gtol=2.23e-16,
        )

        # minus sign super important !
        return -res.x

    def step(self) -> Dict[str, float]:
        """
        Do one BP update: update mu (unless always_zero) then update chi with damping.
        Returns metrics about the chi change.
        """
        D, H = self.cfg.D, self.cfg.H
        damping = self.cfg.damping

        chi_old = self.chi.copy()
        chi_new = self.chi.copy()

        self.mu = self._find_current_mu(self.chi, self.mu)

        for i in range(3):
            f1, f2, f3 = self._assign_f(i)
            for j in range(3):
                second_term = 0.0
                for r in range(H - (i == j), D):
                    c1 = math.comb(D - 1, r)
                    for k in range(D - r):
                        c2 = math.comb(D - 1 - r, k)
                        second_term += (
                            c1 * c2
                            * (self.chi[f1, i] ** r)
                            * (self.chi[f2, i] ** k)
                            * (self.chi[f3, i] ** (D - 1 - r - k))
                        )

                chi_new[i, j] = (
                    damping * np.exp(-(1 / D) * (self.mu[i] + self.mu[j])) * second_term
                    + (1 - damping) * self.chi[i, j]
                )

        chi_new = self._normalize_chi(chi_new)
        self.chi = chi_new

        diff = chi_new - chi_old
        return {
            "chi_diff_max": float(np.max(np.abs(diff))),
            "chi_diff_l2": float(np.linalg.norm(diff)),
            "chi_diff_mean_abs": float(np.mean(np.abs(diff))),
            "chi_min": float(np.min(chi_new)),
            "chi_max": float(np.max(chi_new)),
            "chi_entropy": float(-np.sum(np.where(chi_new > 0, chi_new * np.log(chi_new), 0.0))),
        }

    def compute_Z_node(self) -> float:
        D, H = self.cfg.D, self.cfg.H
        chi = self.chi

        Z_node = 0.0
        for i in range(3):
            f1, f2, f3 = self._assign_f(i)
            for r in range(H, D + 1):
                c1 = math.comb(D, r)
                for k in range(D - r + 1):
                    c2 = math.comb(D - r, k)
                    Z_node += (
                        c1 * c2
                        * (chi[f1, i] ** r)
                        * (chi[f2, i] ** k)
                        * (chi[f3, i] ** (D - r - k))
                    )
        return float(Z_node)

    def compute_Z_edge(self) -> float:
        D = self.cfg.D
        chi = self.chi
        mu = self.mu

        Z_edge = 0.0
        for i in range(3):
            Z_edge += np.exp((2 / D) * mu[i]) * (chi[i, i] ** 2)
            j = (i + 1) % 3
            Z_edge += 2 * np.exp((1 / D) * (mu[i] + mu[j])) * chi[i, j] * chi[j, i]
        return float(Z_edge)

    def compute_m_actual(self, Z_edge: float) -> np.ndarray:
        D = self.cfg.D
        chi = self.chi
        mu = self.mu

        m = np.zeros(3, dtype=float)
        m[0] = (
            np.exp((2 / D) * mu[0]) * chi[0, 0] ** 2
            + np.exp((1 / D) * (mu[0] + mu[1])) * chi[0, 1] * chi[1, 0]
            + np.exp((1 / D) * (mu[0] + mu[2])) * chi[0, 2] * chi[2, 0]
        ) / Z_edge
        m[1] = (
            np.exp((2 / D) * mu[1]) * chi[1, 1] ** 2
            + np.exp((1 / D) * (mu[1] + mu[0])) * chi[1, 0] * chi[0, 1]
            + np.exp((1 / D) * (mu[1] + mu[2])) * chi[1, 2] * chi[2, 1]
        ) / Z_edge
        m[2] = (
            np.exp((2 / D) * mu[2]) * chi[2, 2] ** 2
            + np.exp((1 / D) * (mu[2] + mu[0])) * chi[2, 0] * chi[0, 2]
            + np.exp((1 / D) * (mu[2] + mu[1])) * chi[2, 1] * chi[1, 2]
        ) / Z_edge
        return m

    def compute_quantities(self) -> Dict[str, Any]:
        D = self.cfg.D

        Z_node = self.compute_Z_node()
        Z_edge = self.compute_Z_edge()
        phi_RS = math.log(Z_node) - (D / 2.0) * math.log(Z_edge)

        m_actual = self.compute_m_actual(Z_edge)
        s = phi_RS + float(np.dot(self.mu, m_actual))

        return {
            "Z_node": float(Z_node),
            "Z_edge": float(Z_edge),
            "phi_RS": float(phi_RS),
            "m_actual": m_actual,
            "s": float(s),
        }

    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        last_log_t = t0

        converged = False
        it = 0
        last_metrics: Dict[str, float] = {}

        while it < self.cfg.max_iter:
            last_metrics = self.step()
            diff = last_metrics["chi_diff_max"]

            if it % self.cfg.log_every == 0:
                elapsed = time.time() - t0
                step_dt = time.time() - last_log_t
                last_log_t = time.time()

                q = self.compute_quantities()
                m_err = q["m_actual"] - self.cfg.M_target

                log_payload = {
                    "iter": it,
                    "elapsed_sec": float(elapsed),
                    "sec_since_last_log": float(step_dt),
                    **last_metrics,
                    "mu_0": float(self.mu[0]),
                    "mu_1": float(self.mu[1]),
                    "mu_2": float(self.mu[2]),
                    "m_actual_0": float(q["m_actual"][0]),
                    "m_actual_1": float(q["m_actual"][1]),
                    "m_actual_2": float(q["m_actual"][2]),
                    "total_m_actual": float(np.sum(q["m_actual"])),
                    "chi00": float(self.chi[0, 0]),
                    "chi01": float(self.chi[0, 1]),
                    "chi02": float(self.chi[0, 2]),
                    "chi10": float(self.chi[1, 0]),
                    "chi11": float(self.chi[1, 1]),
                    "chi12": float(self.chi[1, 2]),
                    "chi20": float(self.chi[2, 0]),
                    "chi21": float(self.chi[2, 1]),
                    "chi22": float(self.chi[2, 2]),
                    "m_err_l2": float(np.linalg.norm(m_err)),
                    "m_err_maxabs": float(np.max(np.abs(m_err))),
                    "Z_node": float(q["Z_node"]),
                    "Z_edge": float(q["Z_edge"]),
                    "phi_RS": float(q["phi_RS"]),
                    "s": float(q["s"]),
                    "chi": [float(self.chi[i, j]) for i in range(3) for j in range(3)],
                }

                log_payload["estimated_n_solutions"] = float(np.exp(q["s"] * self.cfg.N))

                if self.cfg.use_wandb and WANDB_AVAILABLE:
                    wandb.log(log_payload, step=it)

            if diff < self.cfg.threshold:
                converged = True
                break

            it += 1

        total_time = time.time() - t0
        return {
            "chi": self.chi,
            "mu": self.mu,
            "iters": it,
            "total_time_sec": float(total_time),
            "converged": bool(converged),
            "last_metrics": last_metrics,
            **self.compute_quantities(),
        }