"""Population dynamics for the 1RSB cavity equation of the H-(dis)assortative
balanced K-partition problem on random d-regular graphs."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Config and result dataclasses
# =============================================================================

@dataclass
class PopDynConfig:
    """Configuration for a population dynamics run."""
    # physical
    K: int
    d: int
    H: int
    problem_type: str = 'assortative'           # 'assortative' or 'disassortative'
    mparisi: float = 1.0                        # Parisi parameter m
    mu: np.ndarray = field(default_factory=lambda: np.array([]))

    # algorithm
    M: int = 10_000                             # population size
    n_iters: int = 5_000                        # iterations
    damping: float = 0.01
    seed: Optional[int] = None

    # initialization of the population
    init_type: str = 'rs_perturb'  # 'rs_perturb', 'rs_exact', 'hard_field', 'almost_unif', 'gaussian', 'manual'
    init_noise: float = 1e-6                    # for 'rs_perturb'
    chi_RS: Optional[np.ndarray] = None         # for 'rs_perturb'/'rs_exact'
    init_population: Optional[np.ndarray] = None  # for 'manual'

    manual_init_chi: Optional[np.ndarray] = None

    # observables:
    n_obs_samples: int = 2_000
    obs_every: int = 100 

    # convergence diagnostics
    track_diff: bool = True

    # logging
    use_wandb: bool = False
    wandb_project: str = 'bp_pop_dyn'
    wandb_group: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_config_extra: Dict[str, Any] = field(default_factory=dict)
    log_every: int = 100
    save_locally: bool = True
    save_dir: str = 'results/pop_dyn'
    run_name: Optional[str] = None

    def __post_init__(self):
        if self.problem_type not in ('assortative', 'disassortative'):
            raise ValueError(
                f"problem_type must be 'assortative' or 'disassortative', "
                f"got {self.problem_type}")

        valid_inits = {'rs_perturb', 'rs_exact', 'hard_field',
                       'almost_unif', 'gaussian', 'manual', "mixed_alpha_hard_manual",}
        if self.init_type not in valid_inits:
            raise ValueError(
                f"init_type must be one of {valid_inits}, got {self.init_type}")

        if self.init_type in ('rs_perturb', 'rs_exact') and self.chi_RS is None:
            raise ValueError(
                f"init_type='{self.init_type}' requires chi_RS to be provided.")
        if self.chi_RS is not None:
            self.chi_RS = np.asarray(self.chi_RS, dtype=float)
            if self.chi_RS.shape != (self.K, self.K):
                raise ValueError(
                    f"chi_RS must have shape ({self.K},{self.K}), got {self.chi_RS.shape}")

        if self.init_type == 'manual' and self.init_population is None:
            raise ValueError("init_type='manual' requires init_population.")
        if self.init_population is not None:
            self.init_population = np.asarray(self.init_population, dtype=float)
            if self.init_population.shape != (self.M, self.K, self.K):
                raise ValueError(
                    f"init_population must have shape ({self.M},{self.K},{self.K}), "
                    f"got {self.init_population.shape}")

        if (self.mu is None
                or (hasattr(self.mu, '__len__') and len(self.mu) == 0)):
            self.mu = np.zeros(self.K, dtype=float)
        else:
            self.mu = np.asarray(self.mu, dtype=float)
            if self.mu.shape != (self.K,):
                raise ValueError(
                    f"mu must have shape ({self.K},), got {self.mu.shape}")

        if not (0.0 <= self.damping < 1.0):
            raise ValueError(f"damping must be in [0, 1), got {self.damping}")

        if self.run_name is None:
            ts = time.strftime('%Y%m%d-%H%M%S')
            self.run_name = (
                f"PD_{self.problem_type[:3]}_K{self.K}_d{self.d}_H{self.H}_"
                f"m{self.mparisi:.2f}_{self.init_type}_{ts}"
            )


@dataclass
class PopDynResult:
    """All outputs of a population dynamics run."""
    # final population
    population: np.ndarray

    # 1RSB observables
    Psi: float
    phi_int: float
    Sigma: float
    log_Zn_mean: float
    log_Ze_mean: float

    # population summary statistics
    mean_chi: np.ndarray
    var_chi: np.ndarray
    spread: float

    # phase classification
    phase: str # 'RS' | 'd1RSB' | 's1RSB' | 'UNSAT'

    # diagnostics
    iters: int
    total_time_sec: float
    history: Dict[str, List[float]]

    config: PopDynConfig

    def to_dict(self) -> Dict[str, Any]:
        out = {
            'Psi': float(self.Psi),
            'phi_int': float(self.phi_int),
            'Sigma': float(self.Sigma),
            'log_Zn_mean': float(self.log_Zn_mean),
            'log_Ze_mean': float(self.log_Ze_mean),
            'mean_chi': self.mean_chi.tolist(),
            'var_chi': self.var_chi.tolist(),
            'spread': float(self.spread),
            'phase': self.phase,
            'iters': int(self.iters),
            'total_time_sec': float(self.total_time_sec),
            'history': self.history,
        }
        cfg = asdict(self.config)

        cfg['mu'] = self.config.mu.tolist()

        if self.config.chi_RS is not None:
            cfg['chi_RS'] = self.config.chi_RS.tolist()

        if self.config.manual_init_chi is not None:
            cfg['manual_init_chi'] = self.config.manual_init_chi.tolist()

        if self.config.init_population is not None:
            cfg['init_population'] = (
                f"<array shape {self.config.init_population.shape}, omitted>")

        out['config'] = cfg

        return out


# =============================================================================
# Helpers
# =============================================================================

def _logmeanexp(log_x: np.ndarray) -> float:
    """log( mean(exp(log_x)) ), where -inf entries contribute 0. Computed stably."""
    finite = np.isfinite(log_x)
    if not finite.any():
        return -math.inf
    n_total = len(log_x)
    x = log_x[finite]
    m = x.max()
    # mean(exp(log_x)) = sum_finite exp(x) / n_total
    return float(m + math.log(np.exp(x - m).sum()) - math.log(n_total))


def _weighted_mean_logweights(log_w: np.ndarray, vals: np.ndarray) -> float:
    """sum_s exp(log_w_s) * vals_s / sum_s exp(log_w_s),  computed stably."""
    finite = np.isfinite(log_w) & np.isfinite(vals)
    if not finite.any():
        return float('nan')
    lw = log_w[finite]
    v = vals[finite]
    m = lw.max()
    w = np.exp(lw - m)
    return float((w * v).sum() / w.sum())

def normalize_chi(chi: np.ndarray) -> np.ndarray:
    s = chi.sum()
    return chi / s if s > 1e-300 else chi


def _normalize_population(pop: np.ndarray) -> np.ndarray:
    sums = pop.sum(axis=(1, 2))
    safe = np.where(sums > 1e-300, sums, 1.0)
    return pop / safe[:, None, None]


def _initialize_population(cfg: PopDynConfig,
                           rng: np.random.Generator) -> np.ndarray:
    """Build the initial population of M messages."""
    M, K = cfg.M, cfg.K

    if cfg.init_type == 'manual':
        return _normalize_population(cfg.init_population.copy())

    if cfg.init_type == 'rs_exact':
        return np.tile(cfg.chi_RS[None, :, :], (M, 1, 1))

    if cfg.init_type == 'rs_perturb':
        pop = np.tile(cfg.chi_RS[None, :, :], (M, 1, 1))
        pop = pop + cfg.init_noise * rng.standard_normal((M, K, K))
        pop = np.maximum(pop, 0.0) 
        return _normalize_population(pop)

    if cfg.init_type == 'hard_field':
        pop = np.zeros((M, K, K))
        a_idx = rng.integers(0, K, size=M)
        b_idx = rng.integers(0, K, size=M)
        pop[np.arange(M), a_idx, b_idx] = 1.0
        return pop

    elif cfg.init_type == "mixed_alpha_hard_manual":
        if cfg.manual_init_chi is None:
            raise ValueError(
                "cfg.manual_init_chi must be provided when "
                "init_type='manual_weighted_hard_fields'"
            )

        chi_manual = np.asarray(cfg.manual_init_chi, dtype=float)

        if chi_manual.shape != (K, K):
            raise ValueError(
                f"manual_init_chi must have shape {(K, K)}, got {chi_manual.shape}"
            )

        if np.any(chi_manual < 0):
            raise ValueError("manual_init_chi must have non-negative entries")

        chi_sum = chi_manual.sum()
        if chi_sum <= 0:
            raise ValueError("manual_init_chi must have positive total mass")

        # Normalize so that chi_manual defines probabilities over the K*K components
        probs = (chi_manual / chi_sum).reshape(-1)

        # Each population message is an almost-hard field:
        # one component is 0.99, all others share 0.01
        low_value = 0.01 / (K * K - 1)
        high_value = 0.99

        pop = np.full((M, K, K), low_value, dtype=float)

        # Sample which component is dominant according to chi_manual
        chosen_components = np.random.choice(K * K, size=M, p=probs)

        a_idx = chosen_components // K
        b_idx = chosen_components % K

        pop[np.arange(M), a_idx, b_idx] = high_value

        return pop

    if cfg.init_type == 'almost_unif':
        pop = np.ones((M, K, K)) + rng.standard_normal((M, K, K)) / 100.0
        pop = np.maximum(pop, 1e-12)
        return _normalize_population(pop)

    if cfg.init_type == 'gaussian':
        pop = rng.random((M, K, K)) + 1.0
        return _normalize_population(pop)

    raise ValueError(f"Unknown init_type {cfg.init_type}")


# =============================================================================
# RS update
# =============================================================================

def _F_update_single(K: int, d: int, H: int, problem_type: str,
                     mu: np.ndarray, parents: np.ndarray
                     ) -> Tuple[np.ndarray, float]:
    """Apply the general RS update F to (d-1) parent messages."""
    chi_unnorm = np.zeros((K, K), dtype=float)
    n_parents = d - 1

    for x in range(K):
        a = parents[:, x, x]
        col_x_sum = parents[:, :, x].sum(axis=1)
        b = col_x_sum - a

        poly = np.array([1.0])
        for al, bl in zip(a, b):
            new_poly = np.zeros(len(poly) + 1)
            new_poly[:-1] += bl * poly
            new_poly[1:]  += al * poly
            poly = new_poly

        if problem_type == 'assortative':
            S_same = poly[max(H - 1, 0):d].sum() if H - 1 < d else 0.0
            S_diff = poly[H:d].sum() if H < d else 0.0
        else:
            S_same = poly[0:max(H - 1, 0)].sum()
            S_diff = poly[0:min(H, d)].sum()

        for y in range(K):
            S = S_same if y == x else S_diff
            chi_unnorm[x, y] = math.exp(-(mu[x] + mu[y]) / d) * S

    Z_F = chi_unnorm.sum()
    if Z_F > 1e-300:
        chi_new = chi_unnorm / Z_F
    else:
        chi_new = np.full((K, K), 1.0 / (K * K))
    return chi_new, float(Z_F)


# =============================================================================
# Observables
# =============================================================================

def _Z_node_from_d_parents(K: int, d: int, H: int, problem_type: str,
                           mu: np.ndarray, parents: np.ndarray) -> float:
    Z = 0.0
    for x in range(K):
        a = parents[:, x, x]
        col_x_sum = parents[:, :, x].sum(axis=1)
        b = col_x_sum - a
        poly = np.array([1.0])
        for al, bl in zip(a, b):
            new_poly = np.zeros(len(poly) + 1)
            new_poly[:-1] += bl * poly
            new_poly[1:]  += al * poly
            poly = new_poly
        if problem_type == 'assortative':
            r_sum = poly[H:d + 1].sum() if H <= d else 0.0
        else:
            r_sum = poly[0:min(H, d + 1)].sum()
        Z += math.exp(-mu[x]) * r_sum
    return float(Z)


def _Z_edge_from_pair(K: int, d: int, mu: np.ndarray,
                      chi_a: np.ndarray, chi_b: np.ndarray) -> float:
    exp_mu = np.exp((mu[:, None] + mu[None, :]) / d)
    W = exp_mu * chi_a.T * chi_b
    return float(W.sum())


# =============================================================================
# PopulationDynamics class
# =============================================================================

class PopulationDynamics:
    """Population dynamics for the 1RSB cavity equation."""

    def __init__(self, config: PopDynConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.population = _initialize_population(config, self.rng)
        self._wandb_run = None

    # ------- one population update step -------

    def step(self) -> float:
        """One population dynamics step."""
        cfg = self.config
        M, K, d = cfg.M, cfg.K, cfg.d
        n_new = max(1, int(np.ceil((1.0 - cfg.damping) * M)))

        prev_mean = self.population.mean(axis=0) if cfg.track_diff else None

        candidates = np.empty((n_new, K, K), dtype=float)
        log_ZF = np.empty(n_new, dtype=float)
        for s in range(n_new):
            idx = self.rng.integers(0, M, size=d - 1)
            parents = self.population[idx]
            chi_new, ZF = _F_update_single(
                K, d, cfg.H, cfg.problem_type, cfg.mu, parents)
            candidates[s] = chi_new
            log_ZF[s] = math.log(ZF) if ZF > 1e-300 else -math.inf

        m = cfg.mparisi
        log_w = m * log_ZF
        finite = np.isfinite(log_w)
        if not finite.any():
            probs = np.full(n_new, 1.0 / n_new)
        else:
            log_w = np.where(finite, log_w, -np.inf)
            log_w_max = log_w[finite].max()
            w = np.exp(log_w - log_w_max)
            w = np.where(finite, w, 0.0)
            tot = w.sum()
            probs = w / tot if tot > 0 else np.full(n_new, 1.0 / n_new)

        chosen = self.rng.choice(n_new, size=n_new, replace=True, p=probs)
        write_idx = self.rng.integers(0, M, size=n_new)
        self.population[write_idx] = candidates[chosen]

        if prev_mean is not None:
            new_mean = self.population.mean(axis=0)
            return float(np.mean(np.abs(new_mean - prev_mean)))
        return 0.0

    # ------- observables -------

    def estimate_observables(self, n_samples: Optional[int] = None) -> Dict[str, float]:
        cfg = self.config
        n = n_samples if n_samples is not None else cfg.n_obs_samples
        M, K, d = cfg.M, cfg.K, cfg.d
        m = cfg.mparisi

        log_Zn = np.empty(n)
        for s in range(n):
            idx = self.rng.integers(0, M, size=d)
            Zn = _Z_node_from_d_parents(
                K, d, cfg.H, cfg.problem_type, cfg.mu, self.population[idx])
            log_Zn[s] = math.log(Zn) if Zn > 1e-300 else -math.inf

        log_Ze = np.empty(n)
        for s in range(n):
            i1, i2 = self.rng.integers(0, M, size=2)
            Ze = _Z_edge_from_pair(K, d, cfg.mu,
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

    # ------- summary stats and phase classification -------

    def population_stats(self) -> Tuple[np.ndarray, np.ndarray, float]:
        mean_chi = self.population.mean(axis=0)
        var_chi = self.population.var(axis=0)
        spread = float(np.sqrt(var_chi).mean())
        return mean_chi, var_chi, spread

    def classify_phase(self, obs: Dict[str, float], spread: float) -> str:
        cfg = self.config
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

    # ------- wandb -------

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

    def _wandb_finish(self, result: PopDynResult):
        if self._wandb_run is None:
            return
        wandb.summary['Psi'] = float(result.Psi)
        wandb.summary['phi_int'] = float(result.phi_int)
        wandb.summary['Sigma'] = float(result.Sigma)
        wandb.summary['spread'] = float(result.spread)
        wandb.summary['phase'] = result.phase
        wandb.summary['iters'] = int(result.iters)
        wandb.summary['total_time_sec'] = float(result.total_time_sec)
        wandb.finish()
        self._wandb_run = None

    # ------- run -------

    def run(self, verbose: int = 0) -> PopDynResult:
        cfg = self.config
        history = {
            'iter': [], 'mean_diff': [],
            'Psi': [], 'phi_int': [], 'Sigma': [],
            'spread': [],
        }

        if cfg.use_wandb:
            self._wandb_init()

        t0 = time.time()
        last_log_t = t0

        for it in range(cfg.n_iters):
            mean_diff = self.step()

            do_log = (it % cfg.log_every == 0) or (it == cfg.n_iters - 1)
            do_obs = (it % cfg.obs_every == 0) or (it == cfg.n_iters - 1)

            if do_obs:
                obs = self.estimate_observables()
                _, _, spread = self.population_stats()
                history['iter'].append(it)
                history['mean_diff'].append(mean_diff)
                history['Psi'].append(obs['Psi'])
                history['phi_int'].append(obs['phi_int'])
                history['Sigma'].append(obs['Sigma'])
                history['spread'].append(spread)

                if verbose >= 2:
                    print(f"  iter {it:>6d}  diff={mean_diff:.2e}  spread={spread:.3e}  "
                          f"Psi={obs['Psi']:+.5f}  phi_int={obs['phi_int']:+.5f}  "
                          f"Sigma={obs['Sigma']:+.5f}")

                if do_log and cfg.use_wandb:
                    payload = {
                        'iter': it,
                        'elapsed_sec': time.time() - t0,
                        'sec_since_last_log': time.time() - last_log_t,
                        'mean_diff': float(mean_diff),
                        'spread': float(spread),
                        'Psi': float(obs['Psi']),
                        'phi_int': float(obs['phi_int']),
                        'Sigma': float(obs['Sigma']),
                        'log_Zn_mean': float(obs['log_Zn_mean']),
                        'log_Ze_mean': float(obs['log_Ze_mean']),
                    }
                    mean_chi, _, _ = self.population_stats()
                    for a in range(cfg.K):
                        for b in range(cfg.K):
                            payload[f'mean_chi_{a}{b}'] = float(mean_chi[a, b])
                    self._wandb_log_step(payload, step=it)
                    last_log_t = time.time()

        final_obs = self.estimate_observables(
            n_samples=max(cfg.n_obs_samples, 5_000))
        mean_chi, var_chi, spread = self.population_stats()
        phase = self.classify_phase(final_obs, spread)

        result = PopDynResult(
            population=self.population.copy(),
            Psi=final_obs['Psi'],
            phi_int=final_obs['phi_int'],
            Sigma=final_obs['Sigma'],
            log_Zn_mean=final_obs['log_Zn_mean'],
            log_Ze_mean=final_obs['log_Ze_mean'],
            mean_chi=mean_chi, var_chi=var_chi, spread=spread,
            phase=phase,
            iters=cfg.n_iters,
            total_time_sec=time.time() - t0,
            history=history,
            config=self.config,
        )

        if verbose >= 1:
            print(f"Done: phase={phase}  spread={spread:.3e}  "
                  f"Psi={result.Psi:+.6f}  phi_int={result.phi_int:+.6f}  "
                  f"Sigma={result.Sigma:+.6f}")

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

    def save(self, result: PopDynResult, folder: Optional[str] = None) -> str:
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


def run_pop_dyn(config: PopDynConfig, verbose: int = 0) -> PopDynResult:
    return PopulationDynamics(config).run(verbose=verbose)


if __name__ == '__main__':
    EXAMPLE = 'mixed_alpha_hard_manual'  # 'rs_stability', 'hard_field', 'sweep_H'
    problem_type='assortative'
    K=2
    Ds=[8]
    Hs=[[5]]
    N_RUNS = 3

    if EXAMPLE == 'rs_stability':
        chi_RS = np.array([
            [
            0.16432256727043526,
            0.08450537733530257,
            0.08450537715893777
            ],
            [
            0.08450538267110654,
            0.16432256610802665,
            0.08450538126375147
            ],
            [
            0.08450538691667205,
            0.08450538568568199,
            0.16432257559008565
            ]
        ])

        SEED = np.random.randint(0, 1000000)
        np.random.seed(SEED)

        for i, D in enumerate(Ds):
            for H in Hs[i]:
                for id_run in range(N_RUNS):
                    pd_cfg = PopDynConfig(
                        K=K, d=D, H=H, problem_type=problem_type,
                        mparisi=1.0,
                        M=10_000, n_iters=2_000, damping=0.9,
                        init_type='rs_perturb',
                        init_noise=1e-6,
                        chi_RS=chi_RS,
                        n_obs_samples=2_000, obs_every=100, log_every=100,
                        seed=SEED,
                        use_wandb=True,
                        wandb_project='bp_pop_dyn',
                        wandb_group='RS_stability',
                        wandb_name=f'{problem_type[:3]}_K{K}_D{D}_H{H}_run{id_run}',
                        save_locally=True,
                        save_dir='results/pop_dyn',
                    )
                    pd_res = run_pop_dyn(pd_cfg)

    elif EXAMPLE == 'hard_field':
        SEED = np.random.randint(0, 1000000)
        np.random.seed(SEED)

        for id_run in range(N_RUNS):
            for i, D in enumerate(Ds):
                for H in Hs[i]:
                    pd_cfg = PopDynConfig(
                        K=K, d=D, H=H, problem_type=problem_type,
                        mparisi=1.0,
                        M=10_000, n_iters=2_000, damping=0.1,
                        init_type='mixed_alpha_hard_manual',
                        n_obs_samples=2_000, obs_every=100, log_every=100,
                        seed=SEED,
                        use_wandb=True,
                        wandb_project='bp_pop_dyn',
                        wandb_group='EXAMPLE',
                        wandb_name=f'{problem_type[:3]}_K{K}_D{D}_H{H}_run{id_run}',
                        save_locally=True,
                        save_dir='results/pop_dyn',
                    )
                    pd_res = run_pop_dyn(pd_cfg)

    elif EXAMPLE == "mixed_alpha_hard_manual":
        problem_type='assortative'
        K=2
        Ds=[8]
        Hs=[[5]]
        N_RUNS = 3
        SEED = np.random.randint(0, 1000000)
        np.random.seed(SEED)

        chi_manual = np.array([
        [
        0.3233764636505374,
        0.19690875827196433
        ],
        [
        0.18513970363509225,
        0.2945750744424062
        ]
        ], dtype=float)

        chi_manual = chi_manual / chi_manual.sum()

        for id_run in range(N_RUNS):
            for i, D in enumerate(Ds):
                for H in Hs[i]:
                    pd_cfg = PopDynConfig(
                        K=K,
                        d=D,
                        H=H,
                        problem_type=problem_type,
                        mparisi=1.0,

                        M=10_000,
                        n_iters=2_000,
                        damping=0.9,

                        init_type="mixed_alpha_hard_manual",
                        manual_init_chi=chi_manual,

                        n_obs_samples=2_000,
                        obs_every=100,
                        log_every=100,

                        seed=SEED,

                        use_wandb=True,
                        wandb_project="bp_pop_dyn",
                        wandb_group="mixed_alpha_hard_manual",
                        wandb_name=f"{problem_type[:3]}_K{K}_D{D}_H{H}_run{id_run}",

                        save_locally=True,
                        save_dir="results/pop_dyn",
                    )

                    pd_res = run_pop_dyn(pd_cfg)

    elif EXAMPLE == 'sweep_H':
        K, d = 3, 7
        problem_type = 'assortative'

        for H in range(2, d):
            bp_res = np.array([
                    [
                    0.12877134742085725,
                    0.0743163452959043,
                    0.0754291990954128
                    ],
                    [
                    0.14856003795673828,
                    0.16143017319274297,
                    0.20808688976193265
                    ],
                    [
                    0.0035178680540969604,
                    0.006689199700180021,
                    0.19319893952213477
                    ]
                ])

            pd_cfg = PopDynConfig(
                K=K, d=d, H=H, problem_type=problem_type,
                mparisi=1.0,
                M=5_000, n_iters=1_000, damping=0.0,
                init_type='rs_perturb',
                init_noise=1e-6, chi_RS=bp_res.chi,
                n_obs_samples=1_000, obs_every=200, log_every=200,
                seed=H, save_locally=True,
                save_dir='results/pop_dyn/sweep',
                wandb_group=f'sweep_K{K}_d{d}',
                use_wandb=False,
            )
            pd_res = run_pop_dyn(pd_cfg)
            print(f"H={H}: BP s={bp_res.s:+.4f}  |  PD phase={pd_res.phase}  "
                  f"spread={pd_res.spread:.2e}  Sigma={pd_res.Sigma:+.4f}")