"""Unified Belief Propagation for H-(dis)assortative balanced K-partitions on
random d-regular graphs."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Config and result dataclasses
# =============================================================================

@dataclass
class BPConfig:
    """Full configuration for a BP run."""
    # physical
    K: int
    d: int
    H: int
    problem_type: str = 'assortative'  # 'assortative' or 'disassortative'
    m_target: np.ndarray = field(default_factory=lambda: np.array([]))

    # algorithm
    max_iter: int = 1_000_000
    threshold: float = 1e-15
    damping: float = 0.01
    init_type: str = 'gaussian'  # 'uniform', 'unif_diag', 'one_hot', 'one_hot_softmax', 'personalized', 'gaussian', 'almost_unif', 'manual'
    init_chi: Optional[np.ndarray] = None
    init_from_folder: Optional[str] = None  # for 'from_folder'
    epsilon: float = 0.5  # used by 'one_hot_softmax' and 'personalized'
    seed: Optional[int] = None

    # mu
    mu_mode: str = 'previous'
    mu_loss: str = 'soft_l1'

    # implementation
    update_path: str = 'fast'  # 'fast' (DP) or 'reference' (nested loops)

    # logging
    use_wandb: bool = False
    wandb_project: str = 'bp_fixed_point'
    wandb_group: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_config_extra: Dict[str, Any] = field(default_factory=dict)
    log_every: int = 1000
    save_locally: bool = True
    save_dir: str = 'results/bp'
    run_name: Optional[str] = None

    def __post_init__(self):
        if self.problem_type not in ('assortative', 'disassortative'):
            raise ValueError(
                f"problem_type must be 'assortative' or 'disassortative', "
                f"got {self.problem_type}")
        if self.update_path not in ('fast', 'reference'):
            raise ValueError(
                f"update_path must be 'fast' or 'reference', got {self.update_path}")
        if self.mu_mode not in ('previous', 'zero', 'always_zero'):
            raise ValueError(
                f"mu_mode must be 'previous', 'zero', or 'always_zero', "
                f"got {self.mu_mode}")

        if (self.m_target is None
                or (hasattr(self.m_target, '__len__') and len(self.m_target) == 0)):
            self.m_target = np.ones(self.K) / self.K
        else:
            self.m_target = np.asarray(self.m_target, dtype=float)
            if self.m_target.shape != (self.K,):
                raise ValueError(
                    f"m_target must have shape ({self.K},), got {self.m_target.shape}")

        if self.init_type == 'manual' and self.init_chi is None:
            raise ValueError("init_type='manual' requires init_chi to be provided.")
        if self.init_type == 'from_folder' and self.init_from_folder is None:
            raise ValueError(
                "init_type='from_folder' requires init_from_folder to be set.")
        if self.init_chi is not None:
            self.init_chi = np.asarray(self.init_chi, dtype=float)
            if self.init_chi.shape != (self.K, self.K):
                raise ValueError(
                    f"init_chi must have shape ({self.K},{self.K}), got {self.init_chi.shape}")

        if self.run_name is None:
            ts = time.strftime('%Y%m%d-%H%M%S')
            self.run_name = (
                f"{self.problem_type[:3]}_K{self.K}_d{self.d}_H{self.H}_"
                f"{self.init_type}_{ts}"
            )

    @classmethod
    def resume_from(cls, folder: str, **overrides) -> "BPConfig":
        """Build a BPConfig that resumes from a previous run's results folder."""
        params_path = os.path.join(folder, 'parameters.json')
        if not os.path.exists(params_path):
            raise FileNotFoundError(
                f"No parameters.json found in {folder}. "
                f"Use init_type='from_folder' + init_from_folder=... directly "
                f"if you only have chi_final.npy.")
        with open(params_path) as f:
            params = json.load(f)

        params.pop('run_name', None)
        params.pop('init_type', None)
        params.pop('init_chi', None)
        params.pop('init_from_folder', None)

        valid = {f.name for f in fields(cls)}
        params = {k: v for k, v in params.items() if k in valid}

        params['init_type'] = 'from_folder'
        params['init_from_folder'] = folder
        params.update(overrides)
        return cls(**params)


@dataclass
class BPResult:
    """All outputs of a BP run."""
    chi: np.ndarray
    mu: np.ndarray
    m_actual: np.ndarray

    Z_node: float
    Z_edge: float
    phi_RS: float
    s: float

    converged: bool
    iters: int
    final_diff: float
    total_time_sec: float

    config: BPConfig

    def to_dict(self) -> Dict[str, Any]:
        out = {
            'chi': self.chi.tolist(),
            'mu': self.mu.tolist(),
            'm_actual': self.m_actual.tolist(),
            'Z_node': float(self.Z_node),
            'Z_edge': float(self.Z_edge),
            'phi_RS': float(self.phi_RS),
            's': float(self.s),
            'converged': bool(self.converged),
            'iters': int(self.iters),
            'final_diff': float(self.final_diff),
            'total_time_sec': float(self.total_time_sec),
        }
        cfg = asdict(self.config)
        cfg['m_target'] = self.config.m_target.tolist()
        if self.config.init_chi is not None:
            cfg['init_chi'] = self.config.init_chi.tolist()
        out['config'] = cfg
        return out


# =============================================================================
# Helpers
# =============================================================================

def assign_f(i: int, K: int) -> List[int]:
    return [(i + s) % K for s in range(K)]


def normalize_chi(chi: np.ndarray) -> np.ndarray:
    s = chi.sum()
    return chi / s if s > 1e-300 else chi


def load_chi_from_folder(folder: str, expected_shape: Optional[Tuple[int, int]] = None
                         ) -> np.ndarray:
    chi_path = os.path.join(folder, 'chi_final.npy')
    if not os.path.exists(chi_path):
        raise FileNotFoundError(f"No chi_final.npy found in {folder}")
    chi = np.load(chi_path)
    if expected_shape is not None and chi.shape != expected_shape:
        raise ValueError(
            f"chi_final.npy in {folder} has shape {chi.shape}, "
            f"expected {expected_shape}")
    return np.asarray(chi, dtype=float)


def initialize_chi(K: int, init_type: str, epsilon: float = 0.5,
                   manual_chi: Optional[np.ndarray] = None,
                   from_folder: Optional[str] = None,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    if init_type == 'manual':
        assert manual_chi is not None
        return normalize_chi(manual_chi.copy())
    if init_type == 'from_folder':
        assert from_folder is not None
        return normalize_chi(load_chi_from_folder(from_folder, expected_shape=(K, K)))

    chi = np.zeros((K, K), dtype=float)
    if init_type == 'uniform':
        chi = np.ones((K, K), dtype=float)
    elif init_type == 'unif_diag':
        chi = np.eye(K) / K
    elif init_type == 'one_hot':
        chi[0, 0] = 1.0
    elif init_type == 'one_hot_softmax':
        chi.fill(epsilon / (K * K - 1))
        chi[0, 0] = 1.0 - epsilon
    elif init_type == 'personalized':
        chi[0, 0] = 1.0 - epsilon
        for a in range(1, K):
            chi[a, a] = epsilon / (K - 1)
    elif init_type == 'gaussian':
        chi = rng.random((K, K)) + 1.0
    elif init_type == 'almost_unif':
        chi = np.ones((K, K)) + np.random.randn(K,K)  / 1.0
    elif init_type == 'almost_unif_std2':
        chi = np.ones((K, K)) + np.random.randn(K,K)  / 2.0
    elif init_type == 'almost_unif_std10':
        chi = np.ones((K, K)) + np.random.randn(K,K)  / 10.0
    else:
        raise ValueError(f"Unknown init_type: {init_type}")
    return normalize_chi(chi)


# =============================================================================
# Mu solver
# =============================================================================

def _density_from_chi_mu(K: int, d: int, mu: np.ndarray, chi: np.ndarray) -> Tuple[np.ndarray, float]:
    W = np.empty((K, K), dtype=float)

    for a in range(K):
        for b in range(K):
            if a == b:
                W[a, b] = (
                    np.exp((2.0 / d) * mu[a])
                    * (chi[a, a] ** 2)
                )
            else:
                W[a, b] = (
                    np.exp((1.0 / d) * (mu[a] + mu[b]))
                    * chi[a, b]
                    * chi[b, a]
                )

    Ze = W.sum()
    m = W.sum(axis=1) / Ze

    return m, float(Ze)


def find_current_mu(K: int, d: int, m_target: np.ndarray, chi: np.ndarray,
                    mu0: np.ndarray, mode: str = 'previous',
                    loss: str = 'soft_l1') -> np.ndarray:
    """Solve for mu so that m(mu, chi) == m_target."""
    if mode == 'zero':
        mu0 = np.zeros(K)

    def residuals(mu):
        m, _ = _density_from_chi_mu(K, d, mu, chi)
        return m - m_target

    res = least_squares(
        residuals, mu0, method='trf', loss=loss,
        xtol=2.23e-16, ftol=2.23e-16, gtol=2.23e-16,
    )

    mu = -res.x  # important: - sign!
    mu = mu - mu.mean()  # remove the gauge mode: sum(mu) = 0
    return mu   
    # return -res.x


# =============================================================================
# Update step
# =============================================================================

def _enumerate_compositions(K_parts: int, total: int):
    """Yield non-negative integer tuples of length K_parts summing to total."""
    if K_parts == 1:
        yield (total,)
        return
    for v in range(total + 1):
        for rest in _enumerate_compositions(K_parts - 1, total - v):
            yield (v,) + rest


def _update_chi_reference(K: int, d: int, H: int, problem_type: str,
                          chi: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Reference: explicit (K-1)-nested-loop sum."""
    chi_new = np.zeros_like(chi)

    for i in range(K):
        f = assign_f(i, K)

        S_self = 0.0
        S_other = 0.0

        for parts in _enumerate_compositions(K, d - 1):
            r1 = parts[0]
            coef = math.factorial(d - 1)
            for p in parts:
                coef //= math.factorial(p)
            term = coef
            for li in range(K):
                term *= chi[f[li], i] ** parts[li]

            if problem_type == 'assortative':
                if r1 >= H - 1:
                    S_self += term
                if r1 >= H:
                    S_other += term
            else:
                if r1 < H - 1:
                    S_self += term
                if r1 < H:
                    S_other += term

        for y in range(K):
            S = S_self if y == i else S_other
            chi_new[i, y] = np.exp(-(1.0 / d) * (mu[i] + mu[y])) * S

    return chi_new


def _update_chi_fast(K: int, d: int, H: int, problem_type: str,
                     chi: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Fast: O(d) per (i, y) using the multinomial theorem."""
    chi_new = np.zeros_like(chi)

    for i in range(K):
        f = assign_f(i, K)
        alpha = chi[f[0], i]                       # "same group" probability
        beta = sum(chi[f[l], i] for l in range(1, K))  # "any other"

        # T[r1] = C(d-1, r1) * alpha^r1 * beta^(d-1-r1)
        T = np.empty(d)
        for r1 in range(d):
            T[r1] = math.comb(d - 1, r1) * (alpha ** r1) * (beta ** (d - 1 - r1))

        if problem_type == 'assortative':
            S_self = T[max(H - 1, 0):d].sum()
            S_other = T[H:d].sum() if H <= d - 1 else 0.0
        else:
            S_self = T[0:max(H - 1, 0)].sum()
            S_other = T[0:min(H, d)].sum()

        for y in range(K):
            S = S_self if y == i else S_other
            chi_new[i, y] = np.exp(-(1.0 / d) * (mu[i] + mu[y])) * S

    return chi_new


# =============================================================================
# Observables
# =============================================================================

def compute_Z_node(K: int, d: int, H: int, problem_type: str,
                   chi: np.ndarray) -> float:
    Z = 0.0
    for i in range(K):
        f = assign_f(i, K)
        alpha = chi[f[0], i]
        beta = sum(chi[f[l], i] for l in range(1, K))

        for r1 in range(d + 1):
            if problem_type == 'assortative' and r1 < H:
                continue
            if problem_type == 'disassortative' and r1 >= H:
                continue
            Z += math.comb(d, r1) * (alpha ** r1) * (beta ** (d - r1))
    return float(Z)


def compute_Z_edge(K: int, d: int, mu: np.ndarray, chi: np.ndarray) -> float:
    Z = 0.0
    for a in range(K):
        Z += np.exp((2.0 / d) * mu[a]) * chi[a, a] ** 2
    for a in range(K):
        for b in range(a + 1, K):
            Z += 2.0 * np.exp((1.0 / d) * (mu[a] + mu[b])) * chi[a, b] * chi[b, a]
    return float(Z)


def compute_phi_RS(d: int, Z_node: float, Z_edge: float) -> float:
    return float(np.log(Z_node) - (d / 2.0) * np.log(Z_edge))


def compute_m_actual(K: int, d: int, mu: np.ndarray,
                     chi: np.ndarray) -> np.ndarray:
    m, _ = _density_from_chi_mu(K, d, mu, chi)
    return m


def compute_entropy(phi_RS: float, mu: np.ndarray, m_actual: np.ndarray) -> float:
    return float(phi_RS + np.dot(mu, m_actual))


def compute_quantities(K: int, d: int, H: int, problem_type: str,
                       chi: np.ndarray, mu: np.ndarray
                       ) -> Tuple[float, float, float, np.ndarray, float]:
    Z_node = compute_Z_node(K, d, H, problem_type, chi)
    Z_edge = compute_Z_edge(K, d, mu, chi)
    phi = compute_phi_RS(d, Z_node, Z_edge)
    m_act = compute_m_actual(K, d, mu, chi)
    s = compute_entropy(phi, mu, m_act)
    return Z_node, Z_edge, phi, m_act, s


# =============================================================================
# BeliefPropagation class
# =============================================================================

class BeliefPropagation:
    """Belief propagation for H-(dis)assortative balanced K-partition problems
    on random d-regular graphs."""

    def __init__(self, config: BPConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        if config.seed is not None:
            np.random.seed(config.seed)

        self.chi = initialize_chi(
            K=config.K,
            init_type=config.init_type,
            epsilon=config.epsilon,
            manual_chi=config.init_chi,
            from_folder=config.init_from_folder,
            rng=self.rng,
        )
        self.mu = np.zeros(config.K)
        self._wandb_run = None

    # ------- update -------

    def _update_chi(self, chi: np.ndarray, mu: np.ndarray) -> np.ndarray:
        if self.config.update_path == 'fast':
            return _update_chi_fast(
                self.config.K, self.config.d, self.config.H,
                self.config.problem_type, chi, mu)
        return _update_chi_reference(
            self.config.K, self.config.d, self.config.H,
            self.config.problem_type, chi, mu)

    def step(self) -> float:
        if self.config.mu_mode == 'always_zero':
            self.mu = np.zeros(self.config.K)
        else:
            self.mu = find_current_mu(
                self.config.K, self.config.d, self.config.m_target,
                self.chi, self.mu, mode=self.config.mu_mode,
                loss=self.config.mu_loss,
            )

        chi_target = self._update_chi(self.chi, self.mu)
        chi_new = (self.config.damping * chi_target
                   + (1.0 - self.config.damping) * self.chi)
        chi_new = normalize_chi(chi_new)
        diff = float(np.max(np.abs(chi_new - self.chi)))
        self.chi = chi_new
        return diff

    # ------- wandb -------

    def _wandb_init(self):
        if not self.config.use_wandb:
            return
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Run `pip install wandb`.")
        cfg_dict = asdict(self.config)
        cfg_dict['m_target'] = self.config.m_target.tolist()
        if self.config.init_chi is not None:
            cfg_dict['init_chi'] = self.config.init_chi.tolist()
        cfg_dict.update(self.config.wandb_config_extra)
        cfg_dict['chi_init'] = self.chi.tolist()
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

    def _wandb_finish(self, result: BPResult):
        if self._wandb_run is None:
            return
        wandb.summary['converged'] = bool(result.converged)
        wandb.summary['iters'] = int(result.iters)
        wandb.summary['total_time_sec'] = float(result.total_time_sec)
        wandb.summary['Z_node'] = float(result.Z_node)
        wandb.summary['Z_edge'] = float(result.Z_edge)
        wandb.summary['phi_RS'] = float(result.phi_RS)
        wandb.summary['s'] = float(result.s)
        for a in range(self.config.K):
            wandb.summary[f'm_actual_{a}'] = float(result.m_actual[a])
            wandb.summary[f'mu_final_{a}'] = float(result.mu[a])
        wandb.finish()
        self._wandb_run = None

    # ------- run -------

    def run(self, verbose: int = 0) -> BPResult:
        K = self.config.K
        log_every = self.config.log_every

        if self.config.use_wandb:
            self._wandb_init()

        t0 = time.time()
        last_log_t = t0
        diff = float('inf')
        it = 0
        converged = False

        for it in range(self.config.max_iter):
            diff = self.step()

            if it % log_every == 0:
                Zn, Ze, phi, m_act, s = compute_quantities(
                    K, self.config.d, self.config.H, self.config.problem_type,
                    self.chi, self.mu)
                if verbose >= 2:
                    print(f"  iter {it:>8d}: diff={diff:.3e}, phi={phi:.6f}, "
                          f"m={np.round(m_act, 4)}")

                payload = {
                    'iter': it,
                    'elapsed_sec': time.time() - t0,
                    'sec_since_last_log': time.time() - last_log_t,
                    'chi_diff_max': diff,
                    'Z_node': float(Zn),
                    'Z_edge': float(Ze),
                    'phi_RS': float(phi),
                    's': float(s),
                    'total_m_actual': float(np.sum(m_act)),
                }
                for a in range(K):
                    payload[f'mu_{a}'] = float(self.mu[a])
                    payload[f'm_actual_{a}'] = float(m_act[a])
                for a in range(K):
                    for b in range(K):
                        payload[f'chi{a}{b}'] = float(self.chi[a, b])

                self._wandb_log_step(payload, step=it)
                last_log_t = time.time()

            if diff < self.config.threshold:
                converged = True
                break

        Zn, Ze, phi, m_act, s = compute_quantities(
            K, self.config.d, self.config.H, self.config.problem_type,
            self.chi, self.mu)

        result = BPResult(
            chi=self.chi.copy(),
            mu=self.mu.copy(),
            m_actual=m_act,
            Z_node=Zn, Z_edge=Ze, phi_RS=phi, s=s,
            converged=converged,
            iters=it + 1,
            final_diff=float(diff),
            total_time_sec=time.time() - t0,
            config=self.config,
        )

        if verbose >= 1:
            print(f"Done: converged={converged}, iters={result.iters}, "
                  f"phi={phi:.6f}, s={s:.6f}, m={np.round(m_act, 6)}")

        if self.config.save_locally:
            self.save(result)

        if self.config.use_wandb:
            self._wandb_finish(result)

        return result

    # ------- persistence -------

    def _result_folder(self) -> str:
        cfg = self.config
        sub = (f"{cfg.problem_type[:3]}_K{cfg.K}_d{cfg.d}_H{cfg.H}/"
               f"{cfg.run_name}")
        return os.path.join(cfg.save_dir, sub)

    def save(self, result: BPResult, folder: Optional[str] = None) -> str:
        if folder is None:
            folder = self._result_folder()
        os.makedirs(folder, exist_ok=True)

        np.save(os.path.join(folder, 'chi_final.npy'), result.chi)
        with open(os.path.join(folder, 'final_results.json'), 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        cfg_dict = asdict(self.config)
        cfg_dict['m_target'] = self.config.m_target.tolist()
        if self.config.init_chi is not None:
            cfg_dict['init_chi'] = self.config.init_chi.tolist()
        with open(os.path.join(folder, 'parameters.json'), 'w') as f:
            json.dump(cfg_dict, f, indent=2)

        return folder

    def load_state(self, folder: str) -> None:
        chi_path = os.path.join(folder, 'chi_final.npy')
        if not os.path.exists(chi_path):
            raise FileNotFoundError(f"No chi_final.npy found in {folder}")
        chi_loaded = np.load(chi_path)
        if chi_loaded.shape != (self.config.K, self.config.K):
            raise ValueError(
                f"Loaded chi has shape {chi_loaded.shape}, "
                f"expected ({self.config.K},{self.config.K})")
        self.chi = chi_loaded


def run_bp(config: BPConfig, verbose: int = 0) -> BPResult:
    return BeliefPropagation(config).run(verbose=verbose)


if __name__ == '__main__':
    problem_type = 'disassortative'
    K = 4
    Ds = [12]
    Hs = [[7, 6, 5, 4, 3, 2, 1]]
    N_RUNS = 3

    # K = 4
    # Ds = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # Hs = [[1, 2], [1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    # N_RUNS = 3

    # K = 5
    # Ds = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # Hs = [[1, 2], [1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    # N_RUNS = 3

    # K = 6
    # Ds = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # Hs = [[1, 2], [1, 2], [1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3, 4]]
    # N_RUNS = 3

    for run_id in range(N_RUNS):
        for i, D in enumerate(Ds):
            for H in Hs[i]:
                SEED = np.random.randint(0, 1000000)
                np.random.seed(SEED)

                cfg = BPConfig(
                    K=K, d=D, H=H,
                    problem_type=problem_type,
                    m_target=np.array([1.0 / K] * K),
                    max_iter=20_000_000,
                    threshold=1e-15,
                    damping=0.01,
                    init_type='almost_unif_std10',
                    mu_mode='previous',
                    seed=SEED,
                    log_every=1000,
                    use_wandb=True,
                    wandb_project='bp_fixed_point',
                    wandb_group='SAVE',
                    wandb_name=f'{problem_type[:3]}_K{K}_D{D}_H{H}_run{run_id}',
                    save_locally=True,
                    save_dir='results/bp',
                )
                res = run_bp(cfg)