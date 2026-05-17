"""Stabilized population dynamics for the 1RSB cavity equation of the
H-(dis)assortative balanced K-partition problem.

This implements the variant described in appendix D of

  C. Koller, F. Behrens, L. Zdeborova, "Counting and hardness-of-finding
  fixed points in cellular automata on random graphs",
  J. Phys. A: Math. Theor. 57 (2024) 465001.

The implementation in `population_dynamics.py` already covers the cavity
update F, the Z^m reweighting and the mu-solver. This file adds the
stability-oriented tricks discussed in the appendix:

  1. Koller-style in-place subpopulation update (fixed subpop indices,
     replacement happens only at those indices).
  2. Distribution-distance convergence diagnostic (per-entry histogram
     TV distance), as a stricter alternative to comparing means.
  3. Two-phase run: a burn-in phase followed by an observable-sampling
     phase that averages observables across many population snapshots
     and reports a standard deviation.
  4. Locked-cluster correction: if (almost) every message in the
     population implies a fixed node value, set phi_int := 0.
  5. Type-II stability check: perturb the population with a small
     non-negative noise and re-iterate; check whether it returns.
  6. BP-weighted hard-field initialization: each message is a pure
     hard-field whose K*K component is sampled from a chi distribution.
  7. Vectorized F update across the L candidates of one step.

The configuration dataclass `PopDynStableConfig` inherits from
`PopDynConfig` and adds the new knobs. See the docstring of that class
for an enumeration.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .population_dynamics import (
        PopDynConfig,
        PopDynResult,
        WANDB_AVAILABLE,
        _logmeanexp,
        _weighted_mean_logweights,
        _normalize_population,
        _compute_magnetization_from_chi,
        _solve_mu_from_chi,
        _initialize_population,
        _F_update_single,
        _Z_node_from_d_parents,
        _Z_edge_from_pair,
    )
except ImportError:
    from population_dynamics import (
        PopDynConfig,
        PopDynResult,
        WANDB_AVAILABLE,
        _logmeanexp,
        _weighted_mean_logweights,
        _normalize_population,
        _compute_magnetization_from_chi,
        _solve_mu_from_chi,
        _initialize_population,
        _F_update_single,
        _Z_node_from_d_parents,
        _Z_edge_from_pair,
    )

try:
    import wandb
except ImportError:
    pass


# =============================================================================
# Config and result dataclasses
# =============================================================================


@dataclass
class PopDynStableConfig(PopDynConfig):
    """Configuration for a stabilized population dynamics run.

    Inherits every field of `PopDynConfig` and adds the stability knobs
    listed below. The class-level defaults of M, damping and max_iter
    are also overridden to track the appendix-D recommendations more
    closely.

    New / overridden parameters (grouped by the stability trick they
    control):

    Koller-style update
    -------------------
    update_scheme : 'koller' or 'legacy'
        'koller' draws the L subpopulation indices *without replacement*
        and overwrites those exact slots after the Z^m reweighting.
        This preserves a fixed (M-L) memory each step and is what the
        appendix prescribes. 'legacy' reproduces the previous code that
        overwrites random indices with replacement.

    Convergence diagnostic
    ----------------------
    convergence_metric : 'histogram', 'moments', 'mean_chi'
        How the per-step diagnostic `mean_diff` (returned by `step()`
        and used by `convergence_threshold` for early stopping) is
        measured.
        * 'histogram' = per-entry total-variation distance between
          the histogram of the previous and current population
          (averaged over the K*K matrix entries). This is the strict
          notion the paper uses ("the distribution of the messages
          ceases to change").
        * 'moments' = sum of L1 changes in mean and variance.
        * 'mean_chi' = L1 change of the mean of chi (the previous
          default, kept for backwards comparability).
    n_hist_bins : int
        Histograms use this many bins on [0, 1] for each chi entry
        (only used when convergence_metric == 'histogram').

    Two-phase run
    -------------
    burn_in_iter : int
        Maximum number of "burn-in" iterations before observables are
        collected. The phase terminates early if
        `convergence_threshold > 0` and the convergence metric drops
        below it.
    n_obs_iters : int
        Number of further iterations during the sampling phase.
    obs_sample_every : int
        Cadence (in iterations) at which observables are sampled and
        added to the running mean/std accumulator.
    n_obs_samples : int (inherited)
        Number of (Zn, Ze) samples per observable estimate. The paper
        uses K = 2*10^8 — typically you raise this for the final
        report.

    Locked-cluster correction
    -------------------------
    hard_field_check : bool
        If True, after sampling we compute the fraction of messages in
        the population that imply a fixed value of one of their two
        nodes (i.e. their x- or y-marginal has a component close to 1).
    hard_field_tol : float
        Tolerance on the marginal: a message is "locked" if
        max(marginal) > 1 - hard_field_tol.
    hard_field_frac_threshold : float
        If the locked fraction is >= this threshold, declare the
        cluster locked and set phi_int := 0 (so Sigma = Psi).
        Set to 1.0 to require *all* messages to be locked; lower it
        if you want to tolerate stragglers caused by numerical noise.

    Type-II stability check
    -----------------------
    stability_check : bool
        If True, after the sampling phase the current population is
        perturbed with |N(0, stability_noise^2)| (and renormalized),
        then iterated for stability_n_iter steps. The distance to the
        unperturbed population is reported as `stability_dist_after`.
    stability_noise : float
        Std of the perturbation. The appendix uses 1e-8 for type-II.
    stability_n_iter : int
        Iterations used for the check.

    Initialization
    --------------
    Adds init_type='hard_field_from_chi': pure hard-field initial
    messages (one component = 1, rest = 0) whose chosen component is
    drawn proportionally to either `manual_init_chi` (if provided) or
    `chi_RS`. This is what the appendix calls "initialized on
    hard-fields proportionally to the obtained message from BP".

    Adds init_type='almost_hard_field_from_chi': same prescription as
    above, but the chosen component holds `init_dominant_mass`
    (default 0.99) and the remaining (1 - init_dominant_mass) is
    spread uniformly over the other K*K - 1 components. Use this when
    pure hard fields collapse the population dynamics to a measure-
    zero fixed point that exact-zero entries cannot escape from; the
    small slack lets the F update transport mass across components.
    `init_dominant_mass` is configurable in (0, 1].

    Overridden defaults
    -------------------
    M = 100_000
    damping = 0.8
    max_iter is auto-computed as burn_in_iter + n_obs_iters and is
    used only for the parent class's bookkeeping; the loop is driven
    by the two phase counts.
    """

    update_scheme: str = 'koller'

    convergence_metric: str = 'histogram'
    n_hist_bins: int = 50

    burn_in_iter: int = 2_000
    n_obs_iters: int = 500
    obs_sample_every: int = 10

    hard_field_check: bool = True
    hard_field_tol: float = 1e-6
    hard_field_frac_threshold: float = 0.95

    stability_check: bool = False
    stability_noise: float = 1e-4
    stability_n_iter: int = 500

    # Dominant-component mass for 'almost_hard_field_from_chi'
    init_dominant_mass: float = 0.99

    # Overridden defaults for the base class
    M: int = 100_000
    damping: float = 0.8
    max_iter: int = 0  # placeholder; auto-computed in __post_init__

    def __post_init__(self):
        # ---- Translate 'hard_field_from_chi' into a 'manual' init ----
        if self.init_type == 'hard_field_from_chi':
            chi_src = (self.manual_init_chi
                       if self.manual_init_chi is not None
                       else self.chi_RS)
            if chi_src is None:
                raise ValueError(
                    "init_type='hard_field_from_chi' requires manual_init_chi "
                    "or chi_RS to be provided.")
            chi_src = np.asarray(chi_src, dtype=float)
            if chi_src.shape != (self.K, self.K):
                raise ValueError(
                    f"chi source for hard_field_from_chi must have shape "
                    f"({self.K},{self.K}), got {chi_src.shape}")
            if np.any(chi_src < 0):
                raise ValueError("chi source must be non-negative")
            s = chi_src.sum()
            if s <= 0:
                raise ValueError("chi source must have positive total mass")

            probs = (chi_src / s).reshape(-1)
            rng = np.random.default_rng(self.seed)
            components = rng.choice(self.K * self.K, size=self.M, p=probs)
            pop = np.zeros((self.M, self.K, self.K), dtype=float)
            a_idx = components // self.K
            b_idx = components % self.K
            pop[np.arange(self.M), a_idx, b_idx] = 1.0
            self.init_population = pop
            self.init_type = 'manual'

        # ---- Translate 'almost_hard_field_from_chi' into a 'manual' init ----
        if self.init_type == 'almost_hard_field_from_chi':
            chi_src = (self.manual_init_chi
                       if self.manual_init_chi is not None
                       else self.chi_RS)
            if chi_src is None:
                raise ValueError(
                    "init_type='almost_hard_field_from_chi' requires "
                    "manual_init_chi or chi_RS to be provided.")
            chi_src = np.asarray(chi_src, dtype=float)
            if chi_src.shape != (self.K, self.K):
                raise ValueError(
                    f"chi source must have shape ({self.K},{self.K}), "
                    f"got {chi_src.shape}")
            if np.any(chi_src < 0):
                raise ValueError("chi source must be non-negative")
            s = chi_src.sum()
            if s <= 0:
                raise ValueError("chi source must have positive total mass")
            if not (0.0 < self.init_dominant_mass <= 1.0):
                raise ValueError(
                    f"init_dominant_mass must be in (0, 1], "
                    f"got {self.init_dominant_mass}")

            high = float(self.init_dominant_mass)
            n_other = self.K * self.K - 1
            low = (1.0 - high) / n_other if n_other > 0 else 0.0
            if n_other == 0:
                high = 1.0

            probs = (chi_src / s).reshape(-1)
            rng = np.random.default_rng(self.seed)
            components = rng.choice(self.K * self.K, size=self.M, p=probs)
            pop = np.full((self.M, self.K, self.K), low, dtype=float)
            a_idx = components // self.K
            b_idx = components % self.K
            pop[np.arange(self.M), a_idx, b_idx] = high
            self.init_population = pop
            self.init_type = 'manual'

        # ---- Ensure parent's max_iter validation passes ----
        if self.max_iter <= 0:
            self.max_iter = max(1, self.burn_in_iter + self.n_obs_iters)

        super().__post_init__()

        # ---- Validate new fields ----
        if self.update_scheme not in {'koller', 'legacy'}:
            raise ValueError(
                f"update_scheme must be 'koller' or 'legacy', "
                f"got {self.update_scheme}")
        if self.convergence_metric not in {'histogram', 'moments', 'mean_chi'}:
            raise ValueError(
                f"convergence_metric must be 'histogram', 'moments' or "
                f"'mean_chi', got {self.convergence_metric}")
        if self.n_hist_bins < 2:
            raise ValueError("n_hist_bins must be >= 2")
        if self.burn_in_iter < 0:
            raise ValueError("burn_in_iter must be >= 0")
        if self.n_obs_iters <= 0:
            raise ValueError("n_obs_iters must be > 0")
        if self.obs_sample_every <= 0:
            raise ValueError("obs_sample_every must be > 0")
        if not (0.0 < self.hard_field_frac_threshold <= 1.0):
            raise ValueError("hard_field_frac_threshold must be in (0, 1]")
        if self.hard_field_tol <= 0:
            raise ValueError("hard_field_tol must be > 0")
        if self.stability_noise <= 0:
            raise ValueError("stability_noise must be > 0")
        if self.stability_n_iter <= 0:
            raise ValueError("stability_n_iter must be > 0")

        # ---- Tag the auto-generated run name as the stable variant ----
        if self.run_name is not None and self.run_name.startswith('PD_'):
            self.run_name = 'PDS_' + self.run_name[len('PD_'):]


@dataclass
class PopDynStableResult(PopDynResult):
    """Adds std-dev observables, locked-cluster info and stability-check info."""
    Psi_std: float = 0.0
    phi_int_std: float = 0.0
    Sigma_std: float = 0.0
    spread_std: float = 0.0
    frac_locked: float = 0.0
    is_locked: bool = False
    stability_passed: Optional[bool] = None
    stability_dist_after: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'Psi_std': float(self.Psi_std),
            'phi_int_std': float(self.phi_int_std),
            'Sigma_std': float(self.Sigma_std),
            'spread_std': float(self.spread_std),
            'frac_locked': float(self.frac_locked),
            'is_locked': bool(self.is_locked),
            'stability_passed': (None if self.stability_passed is None
                                 else bool(self.stability_passed)),
            'stability_dist_after': (None if self.stability_dist_after is None
                                     else float(self.stability_dist_after)),
        })
        return d


# =============================================================================
# Vectorized F update
# =============================================================================


def _F_update_batch(K: int, d: int, H: int, problem_type: str,
                    mu: np.ndarray, parents_batch: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized version of `_F_update_single` over L candidates.

    parents_batch : (L, d-1, K, K)
    returns: candidates (L, K, K) and log_ZF (L,).
    """
    L, n_parents = parents_batch.shape[0], parents_batch.shape[1]
    assert n_parents == d - 1

    chi_unnorm = np.zeros((L, K, K), dtype=float)

    for x in range(K):
        a = parents_batch[:, :, x, x]                    # (L, d-1)
        col_x_sum = parents_batch[:, :, :, x].sum(axis=2)  # (L, d-1)
        b = col_x_sum - a                                  # (L, d-1)

        # Build poly[L, k+1] = prod_{l in parents} (b_l + a_l z)
        poly = np.ones((L, 1), dtype=float)
        for l in range(n_parents):
            al = a[:, l:l + 1]   # (L, 1)
            bl = b[:, l:l + 1]   # (L, 1)
            shifted = al * poly                          # contribution to z^{k+1}
            base = bl * poly                             # contribution to z^k
            new_poly = np.zeros((L, poly.shape[1] + 1), dtype=float)
            new_poly[:, :-1] += base
            new_poly[:, 1:]  += shifted
            poly = new_poly
        # poly shape (L, d)

        if problem_type == 'assortative':
            S_same = poly[:, max(H - 1, 0):d].sum(axis=1)
            S_diff = poly[:, min(H, d):d].sum(axis=1)
        else:
            S_same = poly[:, 0:max(H - 1, 0)].sum(axis=1)
            S_diff = poly[:, 0:min(H, d)].sum(axis=1)

        for y in range(K):
            S = S_same if y == x else S_diff
            chi_unnorm[:, x, y] = math.exp(-(mu[x] + mu[y]) / d) * S

    Z_F = chi_unnorm.sum(axis=(1, 2))                    # (L,)
    bad = Z_F <= 1e-300

    safe_Z = np.where(bad, 1.0, Z_F)
    chi_new = chi_unnorm / safe_Z[:, None, None]
    if bad.any():
        chi_new[bad] = 1.0 / (K * K)

    log_ZF = np.full(L, -np.inf, dtype=float)
    good = ~bad
    if good.any():
        log_ZF[good] = np.log(Z_F[good])

    return chi_new, log_ZF


# =============================================================================
# PopulationDynamicsStable class
# =============================================================================


class PopulationDynamicsStable:

    def __init__(self, config: PopDynStableConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.population = _initialize_population(config, self.rng)
        self.mu = np.asarray(config.mu, dtype=float).copy()
        self._step_count = 0
        self._wandb_run = None

    # ------- mu solver -------

    def _solve_mu(self) -> None:
        cfg = self.config
        n = min(cfg.mu_solver_n_samples, cfg.M)
        idx = self.rng.integers(0, cfg.M, size=n)
        chi_mean = self.population[idx].mean(axis=0)
        s = chi_mean.sum()
        if s > 1e-300:
            chi_mean = chi_mean / s
        self.mu = _solve_mu_from_chi(cfg.K, cfg.d, chi_mean,
                                     cfg.mtarget, self.mu)

    # ------- one population update step -------

    def step(self) -> float:
        cfg = self.config
        M, K, d = cfg.M, cfg.K, cfg.d

        if cfg.enforce_magnetization and (self._step_count % cfg.mu_solver_every == 0):
            self._solve_mu()

        prev_pop = self.population.copy() if cfg.track_diff else None

        L = max(1, int(np.ceil((1.0 - cfg.damping) * M)))

        if cfg.update_scheme == 'koller':
            # Subpopulation indices: drawn WITHOUT replacement, so each
            # update of this step writes to a unique slot. The (M-L)
            # untouched messages provide the population-level memory.
            sub_idx = self.rng.choice(M, size=L, replace=False)
        else:
            sub_idx = self.rng.integers(0, M, size=L)

        # Parents are always sampled from the FULL pre-update population.
        parent_idx = self.rng.integers(0, M, size=(L, d - 1))
        parents_batch = self.population[parent_idx]      # (L, d-1, K, K)

        candidates, log_ZF = _F_update_batch(
            K, d, cfg.H, cfg.problem_type, self.mu, parents_batch)

        # Z^m reweighting at the population level (the 1RSB resampling
        # step from the appendix).
        m = cfg.mparisi
        log_w = m * log_ZF
        finite = np.isfinite(log_w)
        if not finite.any():
            chosen = np.arange(L)
        else:
            lw_max = log_w[finite].max()
            w = np.where(finite, np.exp(log_w - lw_max), 0.0)
            tot = w.sum()
            if tot > 0:
                probs = w / tot
                chosen = self.rng.choice(L, size=L, replace=True, p=probs)
            else:
                chosen = np.arange(L)

        self.population[sub_idx] = candidates[chosen]
        self._step_count += 1

        if prev_pop is not None:
            return self._distribution_distance(prev_pop, self.population)
        return 0.0

    # ------- convergence metric -------

    def _distribution_distance(self, pop_a: np.ndarray,
                               pop_b: np.ndarray) -> float:
        cfg = self.config
        if cfg.convergence_metric == 'mean_chi':
            return float(np.mean(np.abs(pop_a.mean(0) - pop_b.mean(0))))
        if cfg.convergence_metric == 'moments':
            ma, va = pop_a.mean(0), pop_a.var(0)
            mb, vb = pop_b.mean(0), pop_b.var(0)
            return float(np.mean(np.abs(ma - mb))
                         + np.mean(np.abs(va - vb)))
        # histogram (default)
        nb = cfg.n_hist_bins
        total = 0.0
        for aa in range(cfg.K):
            for bb in range(cfg.K):
                ha, _ = np.histogram(pop_a[:, aa, bb], bins=nb,
                                     range=(0.0, 1.0), density=False)
                hb, _ = np.histogram(pop_b[:, aa, bb], bins=nb,
                                     range=(0.0, 1.0), density=False)
                pa = ha / max(cfg.M, 1)
                pb = hb / max(cfg.M, 1)
                total += 0.5 * np.abs(pa - pb).sum()     # TV distance
        return float(total / (cfg.K * cfg.K))

    # ------- observables -------

    def estimate_observables(self, n_samples: Optional[int] = None
                             ) -> Dict[str, float]:
        cfg = self.config
        n = n_samples if n_samples is not None else cfg.n_obs_samples
        M, K, d = cfg.M, cfg.K, cfg.d
        m = cfg.mparisi

        log_Zn = np.empty(n)
        for s in range(n):
            idx = self.rng.integers(0, M, size=d)
            Zn = _Z_node_from_d_parents(
                K, d, cfg.H, cfg.problem_type, self.mu, self.population[idx])
            log_Zn[s] = math.log(Zn) if Zn > 1e-300 else -math.inf

        log_Ze = np.empty(n)
        for s in range(n):
            i1, i2 = self.rng.integers(0, M, size=2)
            Ze = _Z_edge_from_pair(K, d, self.mu,
                                   self.population[i1], self.population[i2])
            log_Ze[s] = math.log(Ze) if Ze > 1e-300 else -math.inf

        log_Zn_mean = _logmeanexp(m * log_Zn)
        log_Ze_mean = _logmeanexp(m * log_Ze)
        Psi = log_Zn_mean - (d / 2.0) * log_Ze_mean

        finite_n = np.isfinite(log_Zn)
        finite_e = np.isfinite(log_Ze)
        E_logZn = _weighted_mean_logweights(m * log_Zn[finite_n],
                                            log_Zn[finite_n])
        E_logZe = _weighted_mean_logweights(m * log_Ze[finite_e],
                                            log_Ze[finite_e])
        phi_int = E_logZn - (d / 2.0) * E_logZe
        Sigma = Psi - m * phi_int

        return {
            'Psi': Psi,
            'phi_int': phi_int,
            'Sigma': Sigma,
            'log_Zn_mean': log_Zn_mean,
            'log_Ze_mean': log_Ze_mean,
        }

    def population_stats(self) -> Tuple[np.ndarray, np.ndarray, float]:
        mean_chi = self.population.mean(axis=0)
        var_chi = self.population.var(axis=0)
        spread = float(np.sqrt(var_chi).mean())
        return mean_chi, var_chi, spread

    def magnetization(self, mean_chi: Optional[np.ndarray] = None) -> np.ndarray:
        """Edge-level magnetization implied by the current population.

        Computes m_b = (sum_a W[a,b] + sum_a W[b,a]) / (2 * sum W), with
        W = exp((mu_a + mu_b)/d) * mean_chi^2-like terms, using the
        helper from the base module.
        """
        cfg = self.config
        if mean_chi is None:
            mean_chi = self.population.mean(axis=0)
        # Normalize defensively, the helper expects a stochastic matrix.
        s = mean_chi.sum()
        if s > 0:
            mean_chi = mean_chi / s
        return _compute_magnetization_from_chi(cfg.K, cfg.d, self.mu, mean_chi)

    def _chi_mag_payload(self, mean_chi: np.ndarray) -> Dict[str, float]:
        """Build a flat dict of mean_chi_ab and m_a entries for wandb."""
        cfg = self.config
        payload: Dict[str, float] = {}
        for a in range(cfg.K):
            for b in range(cfg.K):
                payload[f'mean_chi_{a}{b}'] = float(mean_chi[a, b])
        m = self.magnetization(mean_chi)
        for a in range(cfg.K):
            payload[f'mag_{a}'] = float(m[a])
        return payload

    def fraction_locked(self) -> float:
        """Fraction of messages whose x- or y-marginal is a near-hard-field."""
        cfg = self.config
        tol = cfg.hard_field_tol
        marg_x = self.population.sum(axis=2)
        marg_y = self.population.sum(axis=1)
        locked_x = marg_x.max(axis=1) > 1.0 - tol
        locked_y = marg_y.max(axis=1) > 1.0 - tol
        return float((locked_x | locked_y).mean())

    def classify_phase(self, obs: Dict[str, float], spread: float) -> str:
        is_trivial = spread < 1e-4
        if is_trivial:
            return 'RS'
        Sigma = obs['Sigma']
        Psi = obs['Psi']
        if not math.isfinite(Psi) or Psi < -10:
            return 'UNSAT'
        if Sigma > 1e-3:
            return 'd1RSB'
        if Sigma < -1e-3:
            return 's1RSB'
        return 'd1RSB' if Sigma >= 0 else 's1RSB'

    # ------- Type-II stability check -------

    def check_type_II_stability(self) -> Tuple[bool, float]:
        cfg = self.config
        ref = self.population.copy()
        noise = np.abs(self.rng.normal(0.0, cfg.stability_noise,
                                       size=ref.shape))
        self.population = _normalize_population(ref + noise)
        for _ in range(cfg.stability_n_iter):
            self.step()
        dist = self._distribution_distance(ref, self.population)
        passed = dist < 10.0 * cfg.stability_noise
        return passed, dist

    # ------- wandb plumbing -------

    def _wandb_init(self):
        if not self.config.use_wandb:
            return
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Run `pip install wandb`.")
        cfg_dict = asdict(self.config)
        cfg_dict['mu'] = self.config.mu.tolist()
        if self.config.chi_RS is not None:
            cfg_dict['chi_RS'] = self.config.chi_RS.tolist()
        if self.config.init_population is not None:
            cfg_dict['init_population'] = (
                f"<array shape {self.config.init_population.shape}, omitted>")
        cfg_dict.update(self.config.wandb_config_extra)
        self._wandb_run = wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_name or self.config.run_name,
            group=self.config.wandb_group,
            config=cfg_dict,
        )

    def _wandb_log_step(self, payload: dict, step: int):
        if self._wandb_run is None:
            return
        wandb.log(payload, step=step)

    def _wandb_finish(self, result: PopDynStableResult):
        if self._wandb_run is None:
            return
        wandb.summary['Psi'] = float(result.Psi)
        wandb.summary['Psi_std'] = float(result.Psi_std)
        wandb.summary['phi_int'] = float(result.phi_int)
        wandb.summary['phi_int_std'] = float(result.phi_int_std)
        wandb.summary['Sigma'] = float(result.Sigma)
        wandb.summary['Sigma_std'] = float(result.Sigma_std)
        wandb.summary['spread'] = float(result.spread)
        wandb.summary['spread_std'] = float(result.spread_std)
        wandb.summary['phase'] = result.phase
        wandb.summary['iters'] = int(result.iters)
        wandb.summary['frac_locked'] = float(result.frac_locked)
        wandb.summary['is_locked'] = bool(result.is_locked)
        if result.stability_passed is not None:
            wandb.summary['stability_passed'] = bool(result.stability_passed)
            wandb.summary['stability_dist_after'] = float(result.stability_dist_after)
        wandb.summary['total_time_sec'] = float(result.total_time_sec)
        wandb.summary['converged'] = bool(result.converged)
        for a in range(self.config.K):
            wandb.summary[f'final_mu_{a}'] = float(result.final_mu[a])
        wandb.finish()
        self._wandb_run = None

    # ------- run -------

    def run(self, verbose: int = 0) -> PopDynStableResult:
        cfg = self.config
        history = {
            'iter': [], 'mean_diff': [],
            'Psi': [], 'phi_int': [], 'Sigma': [],
            'spread': [], 'frac_locked': [],
            'mu': [], 'phase_tag': [],   # 0 = burn-in, 1 = sampling
        }

        if cfg.use_wandb:
            self._wandb_init()

        t0 = time.time()
        last_log_t = t0
        converged = False

        # ============== Phase 1: burn-in ==============
        burn_iters_done = 0
        for it in range(cfg.burn_in_iter):
            mean_diff = self.step()

            just_converged = (cfg.convergence_threshold > 0
                              and cfg.track_diff
                              and mean_diff < cfg.convergence_threshold)
            is_last = (it == cfg.burn_in_iter - 1)

            if (it % cfg.log_every == 0) or just_converged or is_last:
                obs = self.estimate_observables()
                _, _, spread = self.population_stats()
                fl = self.fraction_locked()
                history['iter'].append(it)
                history['mean_diff'].append(mean_diff)
                history['Psi'].append(obs['Psi'])
                history['phi_int'].append(obs['phi_int'])
                history['Sigma'].append(obs['Sigma'])
                history['spread'].append(spread)
                history['frac_locked'].append(fl)
                history['mu'].append(self.mu.tolist())
                history['phase_tag'].append(0)

                if verbose >= 2:
                    print(f"  [burn-in] iter {it:>6d}  "
                          f"diff={mean_diff:.2e}  spread={spread:.3e}  "
                          f"Psi={obs['Psi']:+.5f}  Sigma={obs['Sigma']:+.5f}  "
                          f"locked={fl:.2%}")

                if cfg.use_wandb:
                    payload = {
                        'run_phase': 0,
                        'iter': it,
                        'elapsed_sec': time.time() - t0,
                        'sec_since_last_log': time.time() - last_log_t,
                        'mean_diff': float(mean_diff),
                        'spread': float(spread),
                        'frac_locked': float(fl),
                        'Psi': float(obs['Psi']),
                        'phi_int': float(obs['phi_int']),
                        'Sigma': float(obs['Sigma']),
                    }
                    for a in range(cfg.K):
                        payload[f'mu_{a}'] = float(self.mu[a])
                    self._wandb_log_step(payload, step=it)
                    last_log_t = time.time()

            burn_iters_done = it + 1
            if just_converged:
                converged = True
                if verbose >= 1:
                    print(f"Burn-in converged at iter {burn_iters_done}: "
                          f"diff={mean_diff:.2e} < {cfg.convergence_threshold:.2e}")
                break

        # ============== Phase 2: observable sampling ==============
        samples = {
            'Psi': [], 'phi_int': [], 'Sigma': [],
            'log_Zn_mean': [], 'log_Ze_mean': [],
            'spread': [],
        }

        for it in range(cfg.n_obs_iters):
            self.step()
            global_it = burn_iters_done + it

            if it % cfg.obs_sample_every == 0:
                obs = self.estimate_observables()
                _, _, spread = self.population_stats()
                for k in ('Psi', 'phi_int', 'Sigma',
                          'log_Zn_mean', 'log_Ze_mean'):
                    samples[k].append(obs[k])
                samples['spread'].append(spread)

                history['iter'].append(global_it)
                history['mean_diff'].append(float('nan'))
                history['Psi'].append(obs['Psi'])
                history['phi_int'].append(obs['phi_int'])
                history['Sigma'].append(obs['Sigma'])
                history['spread'].append(spread)
                history['frac_locked'].append(self.fraction_locked())
                history['mu'].append(self.mu.tolist())
                history['phase_tag'].append(1)

                if verbose >= 2:
                    print(f"  [sample]  iter {global_it:>6d}  "
                          f"Psi={obs['Psi']:+.5f}  "
                          f"phi_int={obs['phi_int']:+.5f}  "
                          f"Sigma={obs['Sigma']:+.5f}")

                if cfg.use_wandb:
                    payload = {
                        'run_phase': 1,
                        'iter': global_it,
                        'elapsed_sec': time.time() - t0,
                        'spread': float(spread),
                        'Psi': float(obs['Psi']),
                        'phi_int': float(obs['phi_int']),
                        'Sigma': float(obs['Sigma']),
                        'log_Zn_mean': float(obs['log_Zn_mean']),
                        'log_Ze_mean': float(obs['log_Ze_mean']),
                    }
                    self._wandb_log_step(payload, step=global_it)

        # ============== aggregate samples ==============
        def _mean_std(xs):
            arr = np.asarray(xs, dtype=float)
            finite = np.isfinite(arr)
            if not finite.any():
                return float('nan'), float('nan')
            a = arr[finite]
            return (float(a.mean()),
                    float(a.std(ddof=1)) if len(a) > 1 else 0.0)

        Psi, Psi_std = _mean_std(samples['Psi'])
        phi_int, phi_int_std = _mean_std(samples['phi_int'])
        Sigma, Sigma_std = _mean_std(samples['Sigma'])
        log_Zn_mean, _ = _mean_std(samples['log_Zn_mean'])
        log_Ze_mean, _ = _mean_std(samples['log_Ze_mean'])
        spread_mean, spread_std = _mean_std(samples['spread'])

        # ============== locked-cluster correction ==============
        frac_locked = self.fraction_locked()
        is_locked = (cfg.hard_field_check
                     and frac_locked >= cfg.hard_field_frac_threshold)
        if is_locked:
            phi_int = 0.0
            phi_int_std = 0.0
            Sigma = Psi - cfg.mparisi * phi_int
            if verbose >= 1:
                print(f"Population is locked ({frac_locked:.2%} hard-fields); "
                      f"setting phi_int := 0")

        # ============== type-II stability check ==============
        stability_passed = None
        stability_dist_after = None
        if cfg.stability_check:
            if verbose >= 1:
                print(f"Running Type-II stability check "
                      f"(noise={cfg.stability_noise:.2e}, "
                      f"n_iter={cfg.stability_n_iter})...")
            stability_passed, stability_dist_after = self.check_type_II_stability()
            if verbose >= 1:
                print(f"  stability_passed={stability_passed}  "
                      f"dist_after={stability_dist_after:.4e}")

        mean_chi, var_chi, _ = self.population_stats()
        final_obs = dict(Psi=Psi, phi_int=phi_int, Sigma=Sigma,
                         log_Zn_mean=log_Zn_mean, log_Ze_mean=log_Ze_mean)
        phase = self.classify_phase(final_obs, spread_mean)

        result = PopDynStableResult(
            population=self.population.copy(),
            Psi=Psi, phi_int=phi_int, Sigma=Sigma,
            log_Zn_mean=log_Zn_mean, log_Ze_mean=log_Ze_mean,
            mean_chi=mean_chi, var_chi=var_chi, spread=spread_mean,
            phase=phase,
            iters=self._step_count,
            total_time_sec=time.time() - t0,
            history=history,
            converged=converged,
            final_mu=self.mu.copy(),
            config=self.config,
            Psi_std=Psi_std,
            phi_int_std=phi_int_std,
            Sigma_std=Sigma_std,
            spread_std=spread_std,
            frac_locked=frac_locked,
            is_locked=is_locked,
            stability_passed=stability_passed,
            stability_dist_after=stability_dist_after,
        )

        if verbose >= 1:
            print(f"Done: phase={phase}  iters={self._step_count}  "
                  f"spread={spread_mean:.3e}  "
                  f"Psi={Psi:+.6f} +/- {Psi_std:.1e}  "
                  f"phi_int={phi_int:+.6f} +/- {phi_int_std:.1e}  "
                  f"Sigma={Sigma:+.6f} +/- {Sigma_std:.1e}  "
                  f"locked={frac_locked:.2%}")

        if cfg.save_locally:
            self.save(result)
        if cfg.use_wandb:
            self._wandb_finish(result)

        return result

    # ------- persistence -------

    def _result_folder(self) -> str:
        cfg = self.config
        sub = (f"{cfg.problem_type[:3]}_K{cfg.K}_d{cfg.d}_H{cfg.H}_"
               f"m{cfg.mparisi:.2f}/{cfg.run_name}")
        return os.path.join(cfg.save_dir, sub)

    def save(self, result: PopDynStableResult,
             folder: Optional[str] = None) -> str:
        if folder is None:
            folder = self._result_folder()
        os.makedirs(folder, exist_ok=True)

        np.save(os.path.join(folder, 'population_final.npy'), result.population)

        with open(os.path.join(folder, 'final_results.json'), 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        cfg_dict = asdict(self.config)
        cfg_dict['mu'] = self.config.mu.tolist()
        if self.config.chi_RS is not None:
            cfg_dict['chi_RS'] = self.config.chi_RS.tolist()
        if self.config.init_population is not None:
            cfg_dict['init_population'] = (
                f"<array shape {self.config.init_population.shape}, omitted>")
        if self.config.manual_init_chi is not None:
            cfg_dict['manual_init_chi'] = self.config.manual_init_chi.tolist()
        if self.config.mtarget is not None:
            cfg_dict['mtarget'] = self.config.mtarget.tolist()
        cfg_dict['final_mu'] = result.final_mu.tolist()
        with open(os.path.join(folder, 'parameters.json'), 'w') as f:
            json.dump(cfg_dict, f, indent=2)

        return folder

    def load_population(self, folder: str) -> None:
        path = os.path.join(folder, 'population_final.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(f"No population_final.npy in {folder}")
        loaded = np.load(path)
        if loaded.shape != (self.config.M, self.config.K, self.config.K):
            raise ValueError(
                f"Loaded population shape {loaded.shape} does not match "
                f"({self.config.M}, {self.config.K}, {self.config.K})")
        self.population = loaded


def run_pop_dyn_stable(config: PopDynStableConfig,
                       verbose: int = 0) -> PopDynStableResult:
    return PopulationDynamicsStable(config).run(verbose=verbose)


if __name__ == '__main__':
    # =========================================================================
    # TEST_STABILITY: assortative K=2, d=8, H=5, M=10^6, Koller appendix-D
    # settings, uniform-magnetization-enforced, BP-weighted hard-field init.
    # =========================================================================
    problem_type = 'assortative'
    K = 2
    D = 8
    H = 5
    M = 1_000_000
    N_RUNS = 3

    # Almost-hard-field initialization weighted by this chi: the chosen
    # K*K component holds `init_dominant_mass` (0.99) and the rest of
    # the mass is spread uniformly over the other K*K - 1 components.
    # This matches the previous 'mixed_alpha_hard_manual' init but keeps
    # the dominant mass configurable. Pure hard fields collapse the PD
    # to a measure-zero fixed point on this configuration, so we leave
    # a small slack to let the F update transport mass and reveal 1RSB.
    # Replace by the actual BP fixed point for (K=2, d=8, H=5) once known.
    chi_manual = np.array([
    [
      0.4999999999972209,
      0.4999999999937699
    ],
    [
      2.673658504408601e-12,
      6.335535705267995e-12
    ]
    ], dtype=float)
    chi_manual = chi_manual / chi_manual.sum()

    SEED = np.random.randint(0, 1_000_000)
    np.random.seed(SEED)

    for id_run in range(N_RUNS):
        cfg = PopDynStableConfig(
            K=K, d=D, H=H, problem_type=problem_type,
            mparisi=1.0,

            # ---- population ----
            M=M,
            damping=0.8,                          # Koller appendix D

            # ---- init: closest analogue to 'mixed_alpha_hard_manual' ----
            init_type='almost_hard_field_from_chi',
            manual_init_chi=chi_manual,
            init_dominant_mass=0.99,

            # ---- enforce uniform magnetization ----
            enforce_magnetization=True,
            mtarget=np.full(K, 1.0 / K),
            mu_solver_n_samples=M,                # whole population
            mu_solver_every=1,

            # ---- Koller-style two-phase run ----
            burn_in_iter=8_000,                   # Koller burn-in length
            n_obs_iters=2_000,                    # Koller sampling phase length
            obs_sample_every=50,                  # Koller cadence
            # Koller uses K = 2 * 10^8 observable samples per estimate.
            # The current scalar Python loop in estimate_observables
            # makes that intractable on CPU; we use a substantially
            # smaller value but keep the cadence/iter counts identical.
            n_obs_samples=50_000,

            # ---- stability tricks at Koller defaults ----
            update_scheme='koller',
            convergence_metric='histogram',
            n_hist_bins=50,
            convergence_threshold=0.0,            # no early stop; run full burn-in
            hard_field_check=True,
            hard_field_tol=1e-6,
            hard_field_frac_threshold=0.95,
            stability_check=True,
            stability_noise=1e-8,                 # Koller type-II noise
            stability_n_iter=500,

            log_every=100,
            seed=SEED + id_run,

            use_wandb=True,
            wandb_project='bp_pop_dyn',
            wandb_group='TEST_STABILITY',
            wandb_name=f'ass_K{K}_D{D}_H{H}_{id_run}',

            save_locally=True,
            save_dir='results/pop_dyn_stable',
        )

        pd_res = run_pop_dyn_stable(cfg, verbose=1)
