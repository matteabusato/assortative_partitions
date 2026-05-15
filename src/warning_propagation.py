"""Exact deterministic Warning / Survey Propagation for H-(dis)assortative balanced
K-partition problems on random d-regular graphs.

The script:

  1. Iterates the homogeneous SP fixed-point equation

         eta_new(w) = (1/Z) * sum_{w_1,...,w_{d-1}} prod_k eta(w_k)
                              * 1[ w = F(w_1,...,w_{d-1}) ]

     exactly via a DP over sufficient statistics (min_same, max_same, invalid)
     per central color. This is the Parisi parameter m = 0 SP equation and is
     the one used in the literature for "counting WP fixed points" / classifying
     freezing in K-state assortative problems.

  2. At the converged eta, estimates the m-dependent 1RSB observables

         Psi_SP(m) = log Z_n(m) - (d/2) log Z_e(m),
         phi_int(m) = d Psi_SP / d m,
         Sigma(m) = Psi_SP(m) - m * phi_int(m),

     with

         Z_n(m) = E_{w_1,...,w_d ~ eta} [ Z_node(w_1,...,w_d)^m ],
         Z_e(m) = E_{w^(1), w^(2) ~ eta} [ Z_edge(w^(1), w^(2))^m ],

     via Monte Carlo sampling of i.i.d. warning tuples from eta. The same set
     of samples is reused across all m values in a sweep.

Design intentionally mirrors BP.py / population_dynamics.py: dataclass config,
dataclass result, single class, run_wp() helper, argparse main, wandb-safe
logging, results saved under results/wp/.

Single executable file -- no extra modules.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Config and result dataclasses
# =============================================================================

@dataclass
class WPConfig:
    """Full configuration for a Warning / Survey Propagation run."""
    # physical
    K: int
    d: int
    H: int
    problem_type: str = 'assortative'   # 'assortative' or 'disassortative'

    # SP iteration
    eps: float = 1e-3                    # init small-noise reconstruction
    damping: float = 0.85                # eta_next = damping*eta_old + (1-damping)*T(eta_old)
    max_iter: int = 500
    tol: float = 1e-8
    init_type: str = 'small_noise'       # 'small_noise', 'uniform', 'don_t_care'

    # 1RSB observables
    m_values: Optional[List[float]] = None   # if None, defaults to a sweep
    n_obs_samples: int = 20_000          # MC samples for Z_n / Z_e estimation
    obs_seed: Optional[int] = None       # separate seed for obs sampling

    # numerical
    seed: Optional[int] = None
    device: str = 'cpu'
    dtype: str = 'float64'               # 'float64' or 'float32'
    num_threads: Optional[int] = None

    # logging
    use_wandb: bool = False
    wandb_project: str = 'wp_fixed_point'
    wandb_group: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_config_extra: Dict[str, Any] = field(default_factory=dict)
    log_every: int = 10
    save_locally: bool = True
    save_dir: str = 'results/wp'
    run_name: Optional[str] = None
    verbose: int = 1

    def __post_init__(self):
        if self.problem_type not in ('assortative', 'disassortative'):
            raise ValueError(
                f"problem_type must be 'assortative' or 'disassortative', "
                f"got {self.problem_type}")
        if self.init_type not in ('small_noise', 'uniform', 'don_t_care'):
            raise ValueError(
                f"init_type must be one of "
                f"'small_noise','uniform','don_t_care', got {self.init_type}")
        if self.dtype not in ('float64', 'float32'):
            raise ValueError(f"dtype must be 'float64' or 'float32', got {self.dtype}")
        if self.K < 2:
            raise ValueError("K must be >= 2.")
        if self.d < 1:
            raise ValueError("d must be >= 1.")
        if self.K * self.K >= 62:
            raise ValueError(
                "Bit-mask implementation requires K^2 < 62.")
        if not (0.0 <= self.damping < 1.0):
            raise ValueError("damping must satisfy 0 <= damping < 1.")
        if not (0.0 <= self.eps <= 1.0):
            raise ValueError("eps must satisfy 0 <= eps <= 1.")

        if self.m_values is None:
            self.m_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        else:
            self.m_values = [float(m) for m in self.m_values]

        if self.run_name is None:
            ts = time.strftime('%Y%m%d-%H%M%S')
            self.run_name = (
                f"WP_{self.problem_type[:3]}_K{self.K}_d{self.d}_H{self.H}_"
                f"{self.init_type}_{ts}"
            )

    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == 'float64' else torch.float32


@dataclass
class WPResult:
    """All outputs of a WP/SP run."""
    # fixed point
    eta: np.ndarray                      # shape (M,), M = 2^(K^2)
    dont_care_prob: float

    # convergence diagnostics
    converged: bool
    iters: int
    final_residual: float
    support: int
    total_time_sec: float

    # 1RSB observables
    m_values: List[float]
    Psi_SP: List[float]
    phi_int: List[float]
    Sigma: List[float]
    log_Zn_mean: List[float]
    log_Ze_mean: List[float]

    # UNSAT/degeneracy diagnostics for the MC observables
    frac_Zn_zero: float
    frac_Ze_zero: float

    # iteration history
    history: List[Dict[str, float]]

    config: WPConfig

    def to_dict(self) -> Dict[str, Any]:
        def _san(x):
            """Replace nan/inf with None for JSON safety."""
            try:
                v = float(x)
            except Exception:
                return x
            if math.isnan(v) or math.isinf(v):
                return None
            return v

        out = {
            'dont_care_prob': float(self.dont_care_prob),
            'converged': bool(self.converged),
            'iters': int(self.iters),
            'final_residual': float(self.final_residual),
            'support': int(self.support),
            'total_time_sec': float(self.total_time_sec),
            'm_values': list(self.m_values),
            'Psi_SP': [_san(v) for v in self.Psi_SP],
            'phi_int': [_san(v) for v in self.phi_int],
            'Sigma': [_san(v) for v in self.Sigma],
            'log_Zn_mean': [_san(v) for v in self.log_Zn_mean],
            'log_Ze_mean': [_san(v) for v in self.log_Ze_mean],
            'frac_Zn_zero': float(self.frac_Zn_zero),
            'frac_Ze_zero': float(self.frac_Ze_zero),
            'history': self.history,
        }
        cfg = asdict(self.config)
        out['config'] = cfg
        return out


# =============================================================================
# Helpers
# =============================================================================

def _safe_scale_log(m: float, log_x: np.ndarray) -> np.ndarray:
    """Compute m * log_x while keeping -inf entries at -inf.

    NumPy's plain 0 * (-inf) returns NaN (and `np.where` evaluates both
    branches, so a naive `np.where(finite, m*log_x, -inf)` still triggers
    the NaN warning). Compute the product only on the finite mask.
    """
    out = np.full_like(log_x, -np.inf, dtype=np.float64)
    finite = np.isfinite(log_x)
    if finite.any():
        out[finite] = m * log_x[finite]
    return out


def _logmeanexp(log_x: np.ndarray) -> float:
    """log( mean(exp(log_x)) ), handling -inf entries (they contribute 0)."""
    finite = np.isfinite(log_x)
    if not finite.any():
        return -math.inf
    n_total = len(log_x)
    x = log_x[finite]
    m = x.max()
    return float(m + math.log(np.exp(x - m).sum()) - math.log(n_total))


def _weighted_mean_logweights(log_w: np.ndarray, vals: np.ndarray) -> float:
    """sum_s exp(log_w_s) * vals_s / sum_s exp(log_w_s), computed stably.

    Entries with log_w = -inf or vals = -inf are silently dropped (they
    contribute 0 to numerator and denominator under the limiting convention
    used elsewhere in this module).
    """
    finite = np.isfinite(log_w) & np.isfinite(vals)
    if not finite.any():
        return float('nan')
    lw = log_w[finite]
    v = vals[finite]
    m = lw.max()
    w = np.exp(lw - m)
    return float((w * v).sum() / w.sum())


# =============================================================================
# Warning / Survey Propagation core
# =============================================================================

class WarningPropagation:
    """Exact deterministic Survey Propagation (m=0) for K-state assortative /
    disassortative partitions on homogeneous d-regular graphs.

    Notation. A warning is a K x K binary matrix w[c, a] in {0, 1}:

        w[c, a] = 1  iff color c is an allowed value for the incoming neighbor
                     when the central node has color a.

    There are M = 2^(K^2) raw warnings, indexed by their bit-encoding
    p = c*K + a -> bit p of the integer warning id.

    The "don't-care" warning is the all-ones matrix, id = M - 1.

    Damping: eta_next = damping * eta_old + (1 - damping) * T(eta_old),
    with damping close to 1 meaning strong damping.

    Notebook implementation reused unchanged for correctness; only renamed,
    re-organized, and wrapped in the project's standard run/save/wandb plumbing.
    """

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def __init__(self, config: WPConfig):
        self.config = config

        if config.num_threads is not None:
            torch.set_num_threads(int(config.num_threads))

        self.K = int(config.K)
        self.d = int(config.d)
        self.H = int(config.H)
        self.problem_type = config.problem_type
        self.n_in = self.d - 1

        self.device = torch.device(config.device)
        self.dtype = config.torch_dtype()

        self.K2 = self.K * self.K
        self.M = 1 << self.K2
        self.dont_care_id = self.M - 1

        # Bit index p = a*K + b. Same convention as the notebook.
        self.bit_weights = 2 ** torch.arange(
            self.K2, dtype=torch.long, device=self.device)

        ids = torch.arange(self.M, dtype=torch.long, device=self.device)
        bits = (ids[:, None].bitwise_and(self.bit_weights[None, :]) != 0)
        self.warning_bits = bits.reshape(self.M, self.K, self.K)

        self.eye = torch.eye(self.K, dtype=torch.bool, device=self.device)
        self.not_eye = ~self.eye
        self.eye_long = self.eye.to(torch.long)

        # Explicit color-permutation symmetry to enforce balanced sector.
        self.perm_ids = self._build_perm_ids()

        # Precomputed (warning, central-color) -> sufficient statistic effects.
        self.warn_invalid, self.warn_min_inc, self.warn_max_inc = (
            self._precompute_warning_effects()
        )

        # DP state encoding: each color a contributes a digit in base
        # color_state_base, encoding either (min_same, max_same) or invalid.
        self.count_base = self.n_in + 1
        self.invalid_code = self.count_base * self.count_base
        self.color_state_base = self.invalid_code + 1
        self.base_powers = self.color_state_base ** torch.arange(
            self.K, dtype=torch.long, device=self.device)

        self._wandb_run = None

        if config.seed is not None:
            torch.manual_seed(int(config.seed))
            np.random.seed(int(config.seed))

    # ------------------------------------------------------------------
    # Warning representation
    # ------------------------------------------------------------------

    def _matrix_to_ids(self, mats: torch.Tensor) -> torch.Tensor:
        """Convert binary K x K matrices to integer warning ids."""
        flat = mats.reshape(-1, self.K2).to(torch.long)
        return (flat * self.bit_weights[None, :]).sum(dim=-1)

    def warning_matrix(self, warning_id: int) -> torch.Tensor:
        """Return the K x K binary matrix associated with warning_id."""
        return self.warning_bits[warning_id].detach().cpu().to(torch.int64)

    # ------------------------------------------------------------------
    # S_K color-permutation symmetry
    # ------------------------------------------------------------------

    def _build_perm_ids(self) -> torch.Tensor:
        """perm_ids[p, w] = id of (pi.w), with (pi.w)[a,b] = w[pi^-1 a, pi^-1 b]."""
        all_ids = []
        for p in itertools.permutations(range(self.K)):
            inv = [0] * self.K
            for old, new in enumerate(p):
                inv[new] = old
            inv = torch.tensor(inv, dtype=torch.long, device=self.device)
            mats = self.warning_bits[:, inv, :][:, :, inv]
            all_ids.append(self._matrix_to_ids(mats))
        return torch.stack(all_ids, dim=0)

    @torch.no_grad()
    def symmetrize(self, eta: torch.Tensor) -> torch.Tensor:
        """Enforce eta(w) = eta(pi.w) for all pi in S_K."""
        eta = eta.to(device=self.device, dtype=self.dtype)
        out = torch.zeros_like(eta)
        contribution = eta / self.perm_ids.shape[0]
        for pids in self.perm_ids:
            out.index_add_(0, pids, contribution)
        out = torch.clamp(out, min=0.0)
        total = out.sum()
        if total <= 0 or not torch.isfinite(total):
            raise RuntimeError("Invalid survey mass after symmetrization.")
        return out / total

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_eta(self) -> torch.Tensor:
        """Build eta_0 according to config.init_type."""
        cfg = self.config
        if cfg.init_type == 'small_noise':
            return self._init_small_noise(cfg.eps)
        if cfg.init_type == 'uniform':
            return torch.full(
                (self.M,), 1.0 / self.M, dtype=self.dtype, device=self.device)
        if cfg.init_type == 'don_t_care':
            eta = torch.zeros(self.M, dtype=self.dtype, device=self.device)
            eta[self.dont_care_id] = 1.0
            return eta
        raise ValueError(f"Unknown init_type {cfg.init_type}")

    @torch.no_grad()
    def _init_small_noise(self, eps: float) -> torch.Tensor:
        """Small-noise reconstruction initialization.

        eps:   probability of the don't-care warning (all K^2 entries allowed).
        1-eps: distributed uniformly over node-conclusive row warnings
               (fix x_i=a, leave x_j arbitrary): w[c,b] = 1[c=a].
        """
        eta = torch.zeros(self.M, dtype=self.dtype, device=self.device)
        eta[self.dont_care_id] = float(eps)

        row_ids = []
        for a in range(self.K):
            mat = torch.zeros((self.K, self.K), dtype=torch.bool, device=self.device)
            mat[a, :] = True
            wid = self._matrix_to_ids(mat[None, :, :])[0]
            row_ids.append(wid)
        row_ids = torch.stack(row_ids)
        eta[row_ids] = (1.0 - float(eps)) / self.K
        return self.symmetrize(eta)

    # ------------------------------------------------------------------
    # Precomputed warning -> sufficient-statistic effects
    # ------------------------------------------------------------------

    def _precompute_warning_effects(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute, for each (warning w, central color a):

            invalid[w, a]: True if no incoming neighbor color is allowed
            min_inc[w, a]: contribution to MIN possible #same-color neighbors
            max_inc[w, a]: contribution to MAX possible #same-color neighbors
        """
        W = self.warning_bits                                # [M, K, K]
        same_allowed = torch.diagonal(W, dim1=-2, dim2=-1)   # [M, K]
        other_allowed = (W & self.not_eye[None, :, :]).any(dim=-2)  # [M, K]
        invalid = ~(same_allowed | other_allowed)
        min_inc = (same_allowed & ~other_allowed).to(torch.long)
        max_inc = same_allowed.to(torch.long)
        return invalid, min_inc, max_inc

    # ------------------------------------------------------------------
    # DP state encoding / decoding
    # ------------------------------------------------------------------

    def _encode_states(self, min_same: torch.Tensor,
                       max_same: torch.Tensor,
                       invalid: torch.Tensor) -> torch.Tensor:
        """[S, K] -> [S], one integer per state."""
        color_codes = min_same * self.count_base + max_same
        color_codes = torch.where(
            invalid,
            torch.full_like(color_codes, self.invalid_code),
            color_codes,
        )
        return (color_codes * self.base_powers[None, :]).sum(dim=1)

    def _decode_states(self, state_ids: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[S] -> (min_same[S,K], max_same[S,K], invalid[S,K])."""
        color_codes = (state_ids[:, None] // self.base_powers[None, :]
                       ) % self.color_state_base
        invalid = color_codes == self.invalid_code
        valid_codes = torch.where(
            invalid, torch.zeros_like(color_codes), color_codes)
        min_same = valid_codes // self.count_base
        max_same = valid_codes % self.count_base
        return min_same, max_same, invalid

    @torch.no_grad()
    def _state_to_warning_ids(self, state_ids: torch.Tensor) -> torch.Tensor:
        """Map final DP states (over d-1 parents) to outgoing warning ids.

        For outgoing pair (a, b) with a = x_i, b = x_j: neighbor j contributes
        1 same-color neighbor iff b = a.

        assortative:    pair (a, b) allowed iff max_same[a] + 1[b=a] >= H
        disassortative: pair (a, b) allowed iff min_same[a] + 1[b=a] <  H
        """
        min_same, max_same, invalid = self._decode_states(state_ids)
        eq_ab = self.eye_long[None, :, :]                # [1, K, K]

        if self.problem_type == 'assortative':
            allowed = (~invalid[:, :, None]) & (
                max_same[:, :, None] + eq_ab >= self.H)
        else:
            allowed = (~invalid[:, :, None]) & (
                min_same[:, :, None] + eq_ab < self.H)

        return self._matrix_to_ids(allowed)

    # ------------------------------------------------------------------
    # Exact SP update (m = 0)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sp_update_exact(self, eta: torch.Tensor) -> torch.Tensor:
        """One application of T at Parisi m = 0:

            eta_new(w) = (1/Z) sum_{w_1,...,w_{d-1}}
                              prod eta(w_k) * 1[ w = F(w_1,...,w_{d-1}) ].

        The DP iterates over the (d-1) parents, keeping the sufficient
        statistic (min_same[a], max_same[a], invalid[a]) per color. Exact
        because the local constraint depends only on the # of same-color
        neighbors of the central node.
        """
        eta = eta.to(device=self.device, dtype=self.dtype)
        eta = eta / eta.sum()

        support = torch.nonzero(eta > 0, as_tuple=False).flatten()
        support_probs = eta[support]

        # Trivial start state -- before processing any neighbor.
        state_ids = torch.zeros(1, dtype=torch.long, device=self.device)
        state_probs = torch.ones(1, dtype=self.dtype, device=self.device)

        for _ in range(self.n_in):
            min_same, max_same, invalid = self._decode_states(state_ids)

            S = state_ids.numel()
            L = support.numel()

            w_invalid = self.warn_invalid[support]   # [L, K]
            w_min_inc = self.warn_min_inc[support]   # [L, K]
            w_max_inc = self.warn_max_inc[support]   # [L, K]

            new_invalid = invalid[:, None, :] | w_invalid[None, :, :]
            new_min = min_same[:, None, :] + w_min_inc[None, :, :]
            new_max = max_same[:, None, :] + w_max_inc[None, :, :]

            flat_state_ids = self._encode_states(
                new_min.reshape(S * L, self.K),
                new_max.reshape(S * L, self.K),
                new_invalid.reshape(S * L, self.K),
            )
            flat_probs = (
                state_probs[:, None] * support_probs[None, :]
            ).reshape(S * L)

            unique_ids, inverse = torch.unique(
                flat_state_ids, sorted=False, return_inverse=True)

            new_probs = torch.zeros(
                unique_ids.numel(), dtype=self.dtype, device=self.device)
            new_probs.index_add_(0, inverse, flat_probs)

            state_ids = unique_ids
            state_probs = new_probs

        out_ids = self._state_to_warning_ids(state_ids)
        eta_new = torch.zeros(self.M, dtype=self.dtype, device=self.device)
        eta_new.index_add_(0, out_ids, state_probs)
        eta_new = eta_new / eta_new.sum()
        return self.symmetrize(eta_new)

    # ------------------------------------------------------------------
    # Fixed-point solver
    # ------------------------------------------------------------------

    @torch.no_grad()
    def solve(self) -> Tuple[torch.Tensor, List[Dict[str, float]], bool, float, int]:
        """Run damped fixed-point iteration. Returns (eta, history, converged,
        final residual, final support size)."""
        cfg = self.config

        eta = self.init_eta()
        history: List[Dict[str, float]] = []
        residual = float('inf')
        support = int((eta > 1e-14).sum().item())
        it = 0
        converged = False

        for it in range(cfg.max_iter):
            Teta = self.sp_update_exact(eta)
            eta_next = cfg.damping * eta + (1.0 - cfg.damping) * Teta
            eta_next = self.symmetrize(eta_next)

            residual = 0.5 * torch.abs(eta_next - eta).sum().item()
            eta_dc = float(eta_next[self.dont_care_id])
            support = int((eta_next > 1e-14).sum().item())

            history.append({
                'iter': int(it),
                'residual': float(residual),
                'eta_dont_care': float(eta_dc),
                'support': int(support),
            })

            if (it % cfg.log_every == 0 or residual < cfg.tol):
                print(f"  it={it:4d}  residual={residual:.4e}  "
                      f"eta_dc={eta_dc:.12g}  support={support}")
                
                obs_mid = self.estimate_observables(
                    eta_next,
                    cfg.m_values,
                    n_samples=min(2000, cfg.n_obs_samples),   # fewer samples
                    obs_seed=None,
                )
                payload = {f'Sigma_m{m:.2f}': obs_mid['Sigma'][i]
                        for i, m in enumerate(cfg.m_values)}
                payload.update({f'Psi_m{m:.2f}': obs_mid['Psi_SP'][i]
                                for i, m in enumerate(cfg.m_values)})
                self._wandb_log_step(payload, step=it)

            self._wandb_log_step({
                'iter': it,
                'residual': float(residual),
                'eta_dont_care': float(eta_dc),
                'support': int(support),
            }, step=it)

            eta = eta_next

            if residual < cfg.tol:
                converged = True
                break

        return eta, history, converged, float(residual), int(support)

    # ------------------------------------------------------------------
    # Local partition functions Z_node and Z_edge
    # ------------------------------------------------------------------

    def _compute_Z_node_batched(self,
                                warnings_idx: torch.Tensor) -> np.ndarray:
        """Compute Z_node for a batch of d-tuples of warnings.

        warnings_idx: LongTensor of shape [B, d], each row is d sampled
                      warning ids (one tuple).

        Returns: np.ndarray of shape [B], Z_node values.

        Z_node(w_1,...,w_d) = # of (a, c_1,...,c_d) such that
                                 w_k[c_k, a] = 1  for all k,
                              AND assortativity at i:
                                  #{k : c_k = a} >= H   (assortative)
                                  #{k : c_k = a} <  H   (disassortative)

        For fixed central color a, neighbors are independent given a, so
        we count compositions via a small polynomial DP per (sample, a):

            poly[r] = #{(c_1,...,c_d) | exactly r have c_k = a}
                    = product over k of  (a_k + b_k * z)   evaluated as poly,
            where  a_k = w_k[a, a]                       (1 if c_k=a allowed)
                   b_k = sum_{c != a} w_k[c, a]          (# allowed c_k != a)

        Then Z_node = sum_a sum_{r: r in allowed-range} poly[r].
        """
        B, d = warnings_idx.shape
        warnings_idx_cpu = warnings_idx.detach().cpu().numpy()

        K = self.K
        H = self.H
        problem_type = self.problem_type

        # warning_bits_np: [M, K, K] numpy
        warning_bits_np = self.warning_bits.detach().cpu().numpy().astype(np.int64)

        Z_arr = np.zeros(B, dtype=np.float64)

        for s in range(B):
            wids = warnings_idx_cpu[s]
            ws = warning_bits_np[wids]                 # [d, K, K]
            total = 0.0
            for a in range(K):
                # a_k = w_k[a, a],  b_k = sum_{c!=a} w_k[c, a]
                ak = ws[:, a, a].astype(np.float64)
                col_a = ws[:, :, a].sum(axis=1).astype(np.float64)
                bk = col_a - ak

                # Polynomial in z marking "c_k = a".  Degree = d.
                poly = np.array([1.0])
                for k in range(d):
                    new_poly = np.zeros(len(poly) + 1)
                    new_poly[:-1] += bk[k] * poly
                    new_poly[1:]  += ak[k] * poly
                    poly = new_poly

                if problem_type == 'assortative':
                    if H <= d:
                        total += poly[H:d + 1].sum()
                else:
                    total += poly[0:min(H, d + 1)].sum()

            Z_arr[s] = total

        return Z_arr

    def _compute_Z_edge_batched(self,
                                w_left_idx: torch.Tensor,
                                w_right_idx: torch.Tensor) -> np.ndarray:
        """Compute Z_edge for B pairs of warnings.

        For a directed edge (i, j) in the cavity tree, with messages
            w^{i -> j}  (left)   w^{j -> i}  (right),
        Z_edge counts the number of (a, b) = (x_i, x_j) jointly allowed:

            Z_edge = #{(a, b) : w^{i->j}[b, a] = 1  AND  w^{j->i}[a, b] = 1 }.

        Bit convention: w[c, a] means "c is allowed for the SENDER's neighbor
        when the central node has color a". So in w^{i->j}, the central is i;
        the receiver (j) has some color b. The pair (a, b) is allowed by w^{i->j}
        iff a is a consistent center color and b is an allowed neighbor color,
        i.e. w^{i->j}[b, a] = 1. Same logic on the other side.
        """
        wl = self.warning_bits[w_left_idx].detach().cpu().numpy().astype(np.int64)
        wr = self.warning_bits[w_right_idx].detach().cpu().numpy().astype(np.int64)

        # For each pair: sum over (a, b) of wl[b, a] * wr[a, b].
        # That's an einsum-like contraction.
        # wl: [B, K, K], wr: [B, K, K]
        # Z[s] = sum_{a, b} wl[s, b, a] * wr[s, a, b]
        prod = wl.transpose(0, 2, 1) * wr     # wl^T element-wise with wr -> indexed by [s, a, b]
        return prod.reshape(prod.shape[0], -1).sum(axis=1).astype(np.float64)

    # ------------------------------------------------------------------
    # 1RSB observables Psi(m), phi_int(m), Sigma(m)
    # ------------------------------------------------------------------

    def estimate_observables(self,
                             eta: torch.Tensor,
                             m_values: List[float],
                             n_samples: int,
                             obs_seed: Optional[int] = None,
                             ) -> Dict[str, List[float]]:
        """Estimate Psi_SP(m), phi_int(m), Sigma(m) by MC sampling at the
        converged eta.

        For each m:

            Psi_SP(m) = log Z_n(m) - (d/2) log Z_e(m),
            Z_n(m)    = E_{w_1,...,w_d ~ eta}    [Z_node^m],
            Z_e(m)    = E_{w^(1), w^(2) ~ eta}   [Z_edge^m],

            phi_int(m) = d Psi_SP / d m
                       = E[Z_n^m log Z_n] / E[Z_n^m]
                         - (d/2) * E[Z_e^m log Z_e] / E[Z_e^m],

            Sigma(m) = Psi_SP(m) - m * phi_int(m).

        The SAME set of MC samples is reused for all m values: we draw
        n_samples d-tuples for Z_n and n_samples pairs for Z_e.
        """
        eta = eta.detach().cpu()
        # Numpy multinomial sampling (more flexible than torch.multinomial here)
        eta_np = eta.numpy().astype(np.float64)
        eta_np = np.clip(eta_np, 0.0, None)
        s = eta_np.sum()
        if s <= 0 or not np.isfinite(s):
            raise RuntimeError("eta has invalid total mass; cannot sample.")
        eta_np = eta_np / s

        rng = np.random.default_rng(obs_seed)

        # Sample d-tuples for Z_node and 2-tuples for Z_edge.
        node_samples = rng.choice(self.M, size=(n_samples, self.d), p=eta_np)
        edge_samples_l = rng.choice(self.M, size=n_samples, p=eta_np)
        edge_samples_r = rng.choice(self.M, size=n_samples, p=eta_np)

        node_samples_t = torch.from_numpy(node_samples).to(torch.long)
        edge_l_t = torch.from_numpy(edge_samples_l).to(torch.long)
        edge_r_t = torch.from_numpy(edge_samples_r).to(torch.long)

        Z_node_np = self._compute_Z_node_batched(node_samples_t)
        Z_edge_np = self._compute_Z_edge_batched(edge_l_t, edge_r_t)

        # Compute log Z, with -inf for zeros (combinatorial 0 -> excluded).
        log_Zn = np.where(Z_node_np > 0,
                          np.log(np.maximum(Z_node_np, 1e-300)),
                          -np.inf)
        log_Ze = np.where(Z_edge_np > 0,
                          np.log(np.maximum(Z_edge_np, 1e-300)),
                          -np.inf)

        out = {
            'm_values': list(map(float, m_values)),
            'Psi_SP': [],
            'phi_int': [],
            'Sigma': [],
            'log_Zn_mean': [],
            'log_Ze_mean': [],
            'frac_Zn_zero': float(np.mean(Z_node_np <= 0)),
            'frac_Ze_zero': float(np.mean(Z_edge_np <= 0)),
        }

        for m in m_values:
            mlz_n = _safe_scale_log(m, log_Zn)
            mlz_e = _safe_scale_log(m, log_Ze)

            log_Zn_mean = _logmeanexp(mlz_n)
            log_Ze_mean = _logmeanexp(mlz_e)
            Psi = log_Zn_mean - (self.d / 2.0) * log_Ze_mean

            phi_int = (
                _weighted_mean_logweights(mlz_n, log_Zn)
                - (self.d / 2.0)
                * _weighted_mean_logweights(mlz_e, log_Ze)
            )
            Sigma = Psi - m * phi_int

            out['Psi_SP'].append(float(Psi))
            out['phi_int'].append(float(phi_int))
            out['Sigma'].append(float(Sigma))
            out['log_Zn_mean'].append(float(log_Zn_mean))
            out['log_Ze_mean'].append(float(log_Ze_mean))

        return out

    # ------------------------------------------------------------------
    # wandb
    # ------------------------------------------------------------------

    def _wandb_init(self):
        if not self.config.use_wandb:
            return
        if not WANDB_AVAILABLE:
            print("[warning] wandb not available -- continuing without it.")
            return
        try:
            cfg_dict = asdict(self.config)
            cfg_dict.update(self.config.wandb_config_extra)
            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_name or self.config.run_name,
                group=self.config.wandb_group,
                config=cfg_dict,
            )
        except Exception as e:
            print(f"[warning] wandb.init failed: {e}")
            self._wandb_run = None

    def _wandb_log_step(self, payload: dict, step: int):
        if self._wandb_run is None:
            return
        try:
            wandb.log(payload, step=step)
        except Exception:
            pass

    def _wandb_finish(self, result: WPResult):
        if self._wandb_run is None:
            return
        try:
            wandb.summary['converged'] = bool(result.converged)
            wandb.summary['iters'] = int(result.iters)
            wandb.summary['final_residual'] = float(result.final_residual)
            wandb.summary['support'] = int(result.support)
            wandb.summary['total_time_sec'] = float(result.total_time_sec)
            wandb.summary['dont_care_prob'] = float(result.dont_care_prob)
            wandb.summary['frac_Zn_zero'] = float(result.frac_Zn_zero)
            wandb.summary['frac_Ze_zero'] = float(result.frac_Ze_zero)
            for i, m in enumerate(result.m_values):
                wandb.summary[f'Psi_SP_m{m:.3f}'] = float(result.Psi_SP[i])
                wandb.summary[f'phi_int_m{m:.3f}'] = float(result.phi_int[i])
                wandb.summary[f'Sigma_m{m:.3f}'] = float(result.Sigma[i])
            # Sweep tables
            try:
                table = wandb.Table(
                    data=[
                        [m, p, phi, sig]
                        for m, p, phi, sig in zip(
                            result.m_values, result.Psi_SP,
                            result.phi_int, result.Sigma)
                    ],
                    columns=["m", "Psi_SP", "phi_int", "Sigma"],
                )
                wandb.log({
                    "Sigma_vs_m": wandb.plot.line(
                        table, "m", "Sigma", title="Sigma vs m"),
                    "phi_int_vs_m": wandb.plot.line(
                        table, "m", "phi_int", title="phi_int vs m"),
                    "Psi_SP_vs_m": wandb.plot.line(
                        table, "m", "Psi_SP", title="Psi_SP vs m"),
                })
            except Exception:
                pass

            wandb.finish()
        except Exception as e:
            print(f"[warning] wandb.finish failed: {e}")
        finally:
            self._wandb_run = None

    # ------------------------------------------------------------------
    # Top-level driver
    # ------------------------------------------------------------------

    def run(self) -> WPResult:
        cfg = self.config

        if cfg.use_wandb:
            self._wandb_init()

        t0 = time.time()

        # 1. SP fixed point.
        eta, history, converged, residual, support = self.solve()

        # 2. Don't-care probability at the fixed point.
        dc = float(eta[self.dont_care_id].item())

        # 3. 1RSB observables.
        obs = self.estimate_observables(
            eta, cfg.m_values, cfg.n_obs_samples, cfg.obs_seed)

        total_time = time.time() - t0

        result = WPResult(
            eta=eta.detach().cpu().numpy(),
            dont_care_prob=dc,
            converged=converged,
            iters=len(history),
            final_residual=residual,
            support=support,
            total_time_sec=total_time,
            m_values=obs['m_values'],
            Psi_SP=obs['Psi_SP'],
            phi_int=obs['phi_int'],
            Sigma=obs['Sigma'],
            log_Zn_mean=obs['log_Zn_mean'],
            log_Ze_mean=obs['log_Ze_mean'],
            frac_Zn_zero=obs['frac_Zn_zero'],
            frac_Ze_zero=obs['frac_Ze_zero'],
            history=history,
            config=cfg,
        )

        if cfg.verbose >= 1:
            print(
                f"Done: converged={converged}  iters={result.iters}  "
                f"residual={residual:.3e}  support={support}  "
                f"dc={dc:.6f}  time={total_time:.2f}s")
            if result.frac_Zn_zero > 0.5 or result.frac_Ze_zero > 0.5:
                print(
                    f"  [warning] MC samples mostly have Z=0: "
                    f"frac_Zn_zero={result.frac_Zn_zero:.3f}, "
                    f"frac_Ze_zero={result.frac_Ze_zero:.3f}  "
                    "(fixed point likely UNSAT / hard-frozen)")
            for i, m in enumerate(result.m_values):
                print(
                    f"  m={m:.3f}:  Psi={result.Psi_SP[i]:+.6f}  "
                    f"phi_int={result.phi_int[i]:+.6f}  "
                    f"Sigma={result.Sigma[i]:+.6f}")

        if cfg.save_locally:
            self.save(result)
        if cfg.use_wandb:
            self._wandb_finish(result)

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _result_folder(self) -> str:
        cfg = self.config
        sub = (f"{cfg.problem_type[:3]}_K{cfg.K}_d{cfg.d}_H{cfg.H}/"
               f"{cfg.run_name}")
        return os.path.join(cfg.save_dir, sub)

    def save(self, result: WPResult, folder: Optional[str] = None) -> str:
        if folder is None:
            folder = self._result_folder()
        os.makedirs(folder, exist_ok=True)

        # Fixed point.
        np.save(os.path.join(folder, 'eta_final.npy'), result.eta)

        # Full result blob.
        with open(os.path.join(folder, 'final_results.json'), 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        # Parameters (re-saved for resume-style workflows).
        cfg_dict = asdict(self.config)
        with open(os.path.join(folder, 'parameters.json'), 'w') as f:
            json.dump(cfg_dict, f, indent=2)

        # Sigma-vs-m sweep as CSV for easy plotting.
        sweep_csv = os.path.join(folder, 'sigma_vs_m.csv')
        with open(sweep_csv, 'w') as f:
            f.write("m,Psi_SP,phi_int,Sigma,log_Zn_mean,log_Ze_mean\n")
            for i, m in enumerate(result.m_values):
                f.write(
                    f"{m:.6g},{result.Psi_SP[i]:.10g},"
                    f"{result.phi_int[i]:.10g},{result.Sigma[i]:.10g},"
                    f"{result.log_Zn_mean[i]:.10g},"
                    f"{result.log_Ze_mean[i]:.10g}\n")

        # Optional matplotlib plot (purely informational, fails silently).
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            ax[0].plot(result.m_values, result.Psi_SP, 'o-')
            ax[0].set_xlabel('m')
            ax[0].set_ylabel(r'$\Psi_{SP}(m)$')
            ax[0].grid(True, alpha=0.3)
            ax[1].plot(result.m_values, result.phi_int, 'o-')
            ax[1].set_xlabel('m')
            ax[1].set_ylabel(r'$\phi_{int}(m)$')
            ax[1].grid(True, alpha=0.3)
            ax[2].plot(result.m_values, result.Sigma, 'o-')
            ax[2].axhline(0, color='k', lw=0.5)
            ax[2].set_xlabel('m')
            ax[2].set_ylabel(r'$\Sigma(m)$')
            ax[2].grid(True, alpha=0.3)
            fig.suptitle(
                f"WP/SP: {self.config.problem_type} K={self.config.K} "
                f"d={self.config.d} H={self.config.H}")
            fig.tight_layout()
            fig.savefig(os.path.join(folder, 'sigma_vs_m.png'), dpi=120)
            plt.close(fig)
        except Exception:
            pass

        if self.config.verbose >= 1:
            print(f"Saved results in {folder}")

        return folder


# =============================================================================
# Convenience top-level function
# =============================================================================

def run_wp(config: WPConfig) -> WPResult:
    return WarningPropagation(config).run()


# =============================================================================
# CLI
# =============================================================================

def _parse_m_values(s: str) -> List[float]:
    """Parse one of:
        'linspace:0,1,21'  -> np.linspace(0, 1, 21)
        '0.0,0.25,0.5,1.0' -> explicit comma-separated list
    """
    s = s.strip()
    if s.startswith('linspace:'):
        a, b, n = s[len('linspace:'):].split(',')
        return [float(v) for v in np.linspace(float(a), float(b), int(n))]
    return [float(v) for v in s.split(',') if v.strip()]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Warning / Survey Propagation (exact, deterministic) for "
                    "K-state H-(dis)assortative balanced partitions on d-regular graphs.")

    # physical
    p.add_argument('--K', type=int, required=True,
                   help="Number of partition groups.")
    p.add_argument('--d', type=int, required=True,
                   help="Degree of the regular graph.")
    p.add_argument('--H', type=int, required=True,
                   help="Assortativity threshold.")
    p.add_argument('--problem-type', type=str, default='assortative',
                   choices=['assortative', 'disassortative'])

    # SP iteration
    p.add_argument('--eps', type=float, default=1e-3,
                   help="Small-noise reconstruction init: don't-care probability.")
    p.add_argument('--damping', type=float, default=0.85,
                   help="eta_next = damping*eta_old + (1-damping)*T(eta_old).")
    p.add_argument('--max-iter', type=int, default=500)
    p.add_argument('--tol', type=float, default=1e-8,
                   help="Convergence tolerance on L1 residual / 2.")
    p.add_argument('--init-type', type=str, default='small_noise',
                   choices=['small_noise', 'uniform', 'don_t_care'])

    # 1RSB observables sweep
    p.add_argument('--m', type=str, default='linspace:0,1,21',
                   help="Parisi parameter values: comma-separated list OR "
                        "'linspace:a,b,n'.")
    p.add_argument('--n-obs-samples', type=int, default=20_000,
                   help="MC samples for Psi/phi/Sigma estimation.")
    p.add_argument('--obs-seed', type=int, default=None,
                   help="Optional separate seed for observable sampling.")

    # numerics
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--dtype', type=str, default='float64',
                   choices=['float64', 'float32'])
    p.add_argument('--num-threads', type=int, default=None)

    # logging / saving
    p.add_argument('--wandb', action='store_true', dest='use_wandb',
                   help="Enable wandb logging.")
    p.add_argument('--no-wandb', action='store_false', dest='use_wandb',
                   help="Disable wandb logging (default).")
    p.set_defaults(use_wandb=False)
    p.add_argument('--wandb-project', type=str, default='wp_fixed_point')
    p.add_argument('--wandb-group', type=str, default=None)
    p.add_argument('--wandb-name', type=str, default=None)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--save-dir', type=str, default='results/wp')
    p.add_argument('--no-save', action='store_false', dest='save_locally')
    p.set_defaults(save_locally=True)
    p.add_argument('--run-name', type=str, default=None)
    p.add_argument('--verbose', type=int, default=1)
    return p


def config_from_args(args: argparse.Namespace) -> WPConfig:
    return WPConfig(
        K=args.K, d=args.d, H=args.H,
        problem_type=args.problem_type,
        eps=args.eps,
        damping=args.damping,
        max_iter=args.max_iter,
        tol=args.tol,
        init_type=args.init_type,
        m_values=_parse_m_values(args.m),
        n_obs_samples=args.n_obs_samples,
        obs_seed=args.obs_seed,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        num_threads=args.num_threads,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_name=args.wandb_name,
        log_every=args.log_every,
        save_locally=args.save_locally,
        save_dir=args.save_dir,
        run_name=args.run_name,
        verbose=args.verbose,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = build_argparser()
    args = parser.parse_args()
    cfg = config_from_args(args)

    if cfg.verbose >= 1:
        print("=" * 70)
        print("Warning / Survey Propagation")
        print("=" * 70)
        print(f"  problem_type = {cfg.problem_type}")
        print(f"  K = {cfg.K}   d = {cfg.d}   H = {cfg.H}")
        print(f"  init_type = {cfg.init_type}   eps = {cfg.eps}")
        print(f"  damping = {cfg.damping}   max_iter = {cfg.max_iter}   "
              f"tol = {cfg.tol}")
        print(f"  m_values = {cfg.m_values}")
        print(f"  n_obs_samples = {cfg.n_obs_samples}")
        print(f"  seed = {cfg.seed}   device = {cfg.device}   dtype = {cfg.dtype}")
        print(f"  save_dir = {cfg.save_dir}   run_name = {cfg.run_name}")
        print(f"  use_wandb = {cfg.use_wandb}")
        print("=" * 70)

    run_wp(cfg)


if __name__ == '__main__':
    main()