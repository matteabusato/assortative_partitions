import numpy as np
import matplotlib.pyplot as plt
import math
import os
import json
import csv
import time
from scipy import optimize
from scipy.optimize import least_squares

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def assign_f(i):
    return [i, (i + 1) % 3, (i + 2) % 3]


class Factorial:
    def __init__(self, d, H):
        self.d = d
        self.H = H

        size = d - H + 1
        self.mult_chi = np.zeros((size, size))
        for idx_r, r in enumerate(range(H - 1, d)):
            for idx_k, k in enumerate(range(0, d - r)):
                self.mult_chi[idx_r, idx_k] = (
                    math.factorial(d - 1)
                    / (math.factorial(r) * math.factorial(k) * math.factorial(d - 1 - r - k))
                )

        size_n = d - H + 1
        self.mult_znode = np.zeros((size_n, d + 1))
        for idx_r, r in enumerate(range(H, d + 1)):
            for k in range(0, d - r + 1):
                self.mult_znode[idx_r, k] = (
                    math.factorial(d)
                    / (math.factorial(r) * math.factorial(k) * math.factorial(d - r - k))
                )

    def chi_coef(self, r, k):
        return self.mult_chi[r - (self.H - 1), k]

    def znode_coef(self, r, k):
        return self.mult_znode[r - self.H, k]


def normalize(chi):
    if chi.ndim == 2:
        s = chi.sum()
        return chi / s if s > 1e-300 else chi
    s = chi.sum(axis=(-2, -1), keepdims=True)
    s = np.where(s > 1e-300, s, 1.0)
    return chi / s


def initialize_chi_single(init_type='uniform', epsilon=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    chi = np.ones((3, 3))
    if init_type == 'uniform':
        chi = np.ones((3, 3))
    elif init_type == 'unif_diag':
        chi = np.eye(3)
    elif init_type == 'one_hot':
        chi = np.zeros((3, 3))
        chi[0, 0] = 1.0
    elif init_type == 'gaussian':
        chi = rng.random((3, 3)) + 1.0
    elif init_type == 'almost_unif':
        chi = np.ones((3, 3)) + rng.standard_normal((3, 3)) / 100.0
    return normalize(chi)


def find_mu_from_target(d, m_target, chi_mean, mu0=None, loss='soft_l1'):
    if mu0 is None:
        mu0 = np.zeros(3)

    def m_values(mu):
        den = 0.0
        for a in range(3):
            den += np.exp((2.0 / d) * mu[a]) * chi_mean[a, a] ** 2
        for (a, b) in [(0, 1), (1, 2), (2, 0)]:
            den += 2.0 * np.exp((1.0 / d) * (mu[a] + mu[b])) * chi_mean[a, b] * chi_mean[b, a]

        out = np.zeros(3)
        for a in range(3):
            num = np.exp((2.0 / d) * mu[a]) * chi_mean[a, a] ** 2
            for b in range(3):
                if b != a:
                    num += np.exp((1.0 / d) * (mu[a] + mu[b])) * chi_mean[a, b] * chi_mean[b, a]
            out[a] = num / den
        return out

    res = least_squares(
        lambda mu: m_values(mu) - m_target,
        mu0, method='trf', loss=loss,
        xtol=2.23e-16, ftol=2.23e-16, gtol=2.23e-16,
    )
    return -res.x


class population_dynamics_3ass:
    def __init__(
        self,
        d, H,
        m_parisi=1.0,
        mu=None,
        m_target=None,
        M=10000,
        num_samples=100000,
        damping=0.8,
        init_type='hard_fields',
        init_chi_bp=None,
        init_noise=1e-3,
        mu_update='fixed',         # 'fixed' or 'target_m'
        mu_solver_loss='soft_l1',
        impose_symmetry=False,
        max_iter=10000,
        tol=10.0,
        convergence_check_interval=200,
        sampling_threshold=8000,
        sampling_interval=50,
        m_parisi_list=None,
        seed=None,
        use_wandb=False,
        wandb_project='pop_dyn_3ass',
        wandb_group=None,
        wandb_name=None,
        wandb_config_extra=None,
        log_every=100,
        log_histograms_every=1000,
    ):
        self.d = d
        self.H = H
        self.m_parisi = m_parisi
        self.mu = np.zeros(3) if mu is None else np.asarray(mu, dtype=float)
        self.m_target = None if m_target is None else np.asarray(m_target, dtype=float)
        self.M = M
        self.num_samples = num_samples
        self.damping = damping
        self.init_type = init_type
        self.init_chi_bp = init_chi_bp
        self.init_noise = init_noise
        self.mu_update = mu_update
        self.mu_solver_loss = mu_solver_loss
        self.impose_symmetry = impose_symmetry
        self.max_iter = max_iter
        self.tol = tol
        self.convergence_check_interval = convergence_check_interval
        self.sampling_threshold = sampling_threshold
        self.sampling_interval = sampling_interval
        self.m_parisi_list = (np.linspace(1e-5, 1, 30) if m_parisi_list is None
                              else np.asarray(m_parisi_list))
        self.seed = seed

        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_group = wandb_group
        self.wandb_name = wandb_name
        self.wandb_config_extra = wandb_config_extra or {}
        self.log_every = log_every
        self.log_histograms_every = log_histograms_every
        self._wandb_run = None
        self._global_step = 0

        if self.use_wandb and not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Run `pip install wandb`.")

        self.rng = np.random.default_rng(seed)

        self.FACT = Factorial(d, H)
        self.nb_new = max(1, round((1 - self.damping) * self.M))

        self._fmap = np.array([assign_f(i) for i in range(3)], dtype=int)

        self.population = np.zeros((M, 3, 3))
        self._initialize_population()
        self.old_population = self.population.copy()

        self.no_update = False

        self.psi = self.phi = self.complexity = None
        self.rho = None
        self.psi_mean = self.phi_mean = self.complexity_mean = None
        self.rho_mean = None
        self.psi_std = self.phi_std = self.complexity_std = None
        self.rho_std = None

        self.psi_list = self.psi_list_std = None
        self.phi_list = self.phi_list_std = None
        self.complexity_list = self.complexity_list_std = None
        self.rho_list = self.rho_list_std = None

        self.fitting_param_phi = None
        self.phi_s = self.phi_d = None

    def __repr__(self):
        s = ("population_dynamics_3ass\n"
             f"  d={self.d}, H={self.H}\n"
             f"  Parisi m={self.m_parisi}, μ={self.mu}, M={self.M}\n"
             f"  init_type={self.init_type}, mu_update={self.mu_update}\n")
        if self.phi_mean is not None:
            s += (f"  Ψ = {self.psi_mean:.6f} ± {self.psi_std:.2e}\n"
                  f"  φ = {self.phi_mean:.6f} ± {self.phi_std:.2e}\n"
                  f"  Σ = {self.complexity_mean:.6f} ± {self.complexity_std:.2e}\n"
                  f"  ρ = {self.rho_mean} ± {self.rho_std}\n")
        if self.phi_s is not None:
            s += f"  φ_s = {self.phi_s}, φ_d = {self.phi_d}\n"
        return s


    def _wandb_config(self):
        cfg = {
            'd': self.d, 'H': self.H,
            'm_parisi': float(self.m_parisi),
            'mu_init': self.mu.tolist(),
            'm_target': None if self.m_target is None else self.m_target.tolist(),
            'M': self.M, 'num_samples': self.num_samples,
            'damping': self.damping,
            'init_type': self.init_type,
            'init_noise': self.init_noise,
            'mu_update': self.mu_update,
            'mu_solver_loss': self.mu_solver_loss,
            'impose_symmetry': self.impose_symmetry,
            'max_iter': self.max_iter, 'tol': self.tol,
            'convergence_check_interval': self.convergence_check_interval,
            'sampling_threshold': self.sampling_threshold,
            'sampling_interval': self.sampling_interval,
            'm_parisi_list': self.m_parisi_list.tolist(),
            'seed': self.seed,
            'log_every': self.log_every,
            'log_histograms_every': self.log_histograms_every,
        }
        cfg.update(self.wandb_config_extra)
        return cfg

    def wandb_init(self, name=None, group=None, config_extra=None, reinit=False):
        if not self.use_wandb:
            return None
        cfg = self._wandb_config()
        if config_extra:
            cfg.update(config_extra)
        self._wandb_run = wandb.init(
            project=self.wandb_project,
            name=name or self.wandb_name or f"d{self.d}_H{self.H}_m{self.m_parisi:.3f}",
            group=group or self.wandb_group,
            config=cfg,
            reinit=reinit,
        )
        return self._wandb_run

    def wandb_finish(self):
        if self._wandb_run is not None:
            wandb.finish()
            self._wandb_run = None

    def _wandb_log(self, payload, step=None, commit=True):
        if self._wandb_run is None:
            return
        wandb.log(payload, step=step, commit=commit)

    def _log_step_metrics(self, it, t_elapsed, diff_val=None,
                          include_observables=False, include_histograms=False):
        if self._wandb_run is None:
            return

        chi_mean = self.population.mean(axis=0)
        chi_std = self.population.std(axis=0)

        payload = {
            'iter': it,
            'global_step': self._global_step,
            'elapsed_sec': t_elapsed,
            'no_update': int(self.no_update),
            'm_parisi': float(self.m_parisi),
        }

        for a in range(3):
            payload[f'mu_{a}'] = float(self.mu[a])

        for i in range(3):
            for j in range(3):
                payload[f'chi_mean_{i}{j}'] = float(chi_mean[i, j])
                payload[f'chi_std_{i}{j}'] = float(chi_std[i, j])

        try:
            d = self.d
            den = 0.0
            for a in range(3):
                den += np.exp((2.0 / d) * self.mu[a]) * chi_mean[a, a] ** 2
            for (a, b) in [(0, 1), (1, 2), (2, 0)]:
                den += 2.0 * np.exp((1.0 / d) * (self.mu[a] + self.mu[b])) * chi_mean[a, b] * chi_mean[b, a]
            if den > 0:
                for a in range(3):
                    num = np.exp((2.0 / d) * self.mu[a]) * chi_mean[a, a] ** 2
                    for b in range(3):
                        if b != a:
                            num += np.exp((1.0 / d) * (self.mu[a] + self.mu[b])) * chi_mean[a, b] * chi_mean[b, a]
                    payload[f'm_chi_mean_{a}'] = float(num / den)
        except Exception:
            pass

        # hard-field fraction: messages with one entry ≈ 1
        max_entry = self.population.max(axis=(1, 2))
        payload['frac_hard_fields'] = float((max_entry > 0.99).mean())
        payload['chi_max_mean'] = float(max_entry.mean())

        if diff_val is not None:
            payload['diff'] = float(diff_val)

        if include_observables and self.psi is not None and not np.isnan(self.psi):
            payload['psi'] = float(self.psi)
            payload['phi'] = float(self.phi)
            payload['complexity'] = float(self.complexity)
            if self.rho is not None:
                for a in range(3):
                    payload[f'rho_{a}'] = float(self.rho[a])

        if include_histograms:
            for i in range(3):
                for j in range(3):
                    try:
                        payload[f'hist_chi_{i}{j}'] = wandb.Histogram(
                            self.population[:, i, j], num_bins=64)
                    except Exception:
                        pass

        self._wandb_log(payload, step=self._global_step, commit=True)


    def _initialize_population(self):
        M = self.M
        if self.init_type == 'hard_fields':
            self.population = np.zeros((M, 3, 3))
            idx = self.rng.integers(0, 9, size=M)
            rows, cols = idx // 3, idx % 3
            self.population[np.arange(M), rows, cols] = 1.0

        elif self.init_type == 'bp_fixed_point':
            if self.init_chi_bp is None:
                raise ValueError("init_chi_bp must be provided for 'bp_fixed_point'")
            self.population = np.broadcast_to(
                self.init_chi_bp, (M, 3, 3)).copy()

        elif self.init_type == 'bp_fixed_point_noisy':
            if self.init_chi_bp is None:
                raise ValueError("init_chi_bp must be provided for 'bp_fixed_point_noisy'")
            base = np.broadcast_to(self.init_chi_bp, (M, 3, 3)).copy()
            noise = self.rng.normal(0.0, self.init_noise, size=(M, 3, 3))
            self.population = np.maximum(base + noise, 0.0)

        elif self.init_type == 'uniform':
            self.population = np.ones((M, 3, 3))

        elif self.init_type == 'almost_unif':
            self.population = np.ones((M, 3, 3)) + self.rng.standard_normal((M, 3, 3)) / 100.0

        elif self.init_type == 'random':
            self.population = self.rng.random((M, 3, 3))

        else:
            raise ValueError(f"Unknown init_type: {self.init_type}")

        self.population = normalize(self.population)

    def reset_population(self):
        self._initialize_population()
        self.old_population = self.population.copy()


    def _chi_update_batch(self, parents):
        n_new = parents.shape[0]
        d = self.d
        H = self.H
        new_chi = np.zeros((n_new, 3, 3))

        for x in range(3):
            f1, f2, f3 = self._fmap[x]

            a = parents[:, :, f1, x]
            b = parents[:, :, f2, x]
            c = parents[:, :, f3, x]

            dp = np.zeros((n_new, d))
            dp[:, 0] = 1.0
            for l in range(d - 1):
                a_l = a[:, l]
                bc_l = b[:, l] + c[:, l]
                new_dp = dp * bc_l[:, None]
                new_dp[:, 1:] += dp[:, :-1] * a_l[:, None]
                dp = new_dp

            for y in range(3):
                r_min = H - (1 if y == x else 0)
                if r_min >= d:
                    contrib = np.zeros(n_new)
                else:
                    r_min_eff = max(r_min, 0)
                    contrib = dp[:, r_min_eff:d].sum(axis=1)
                factor = np.exp(-(self.mu[x] + self.mu[y]) / d)
                new_chi[:, x, y] = factor * contrib

        Z = new_chi.sum(axis=(1, 2))
        safe_Z = np.where(Z > 1e-300, Z, 1.0)
        new_chi = new_chi / safe_Z[:, None, None]
        return new_chi, Z

    def step(self):
        if not self.no_update:
            self.old_population = self.population.copy()

        if self.mu_update == 'target_m' and self.m_target is not None:
            chi_mean = self.population.mean(axis=0)
            try:
                self.mu = find_mu_from_target(
                    self.d, self.m_target, chi_mean,
                    mu0=self.mu, loss=self.mu_solver_loss,
                )
            except Exception:
                pass

        n_new = self.nb_new
        idx = self.rng.integers(0, self.M, size=n_new * (self.d - 1))
        parents = self.population[idx].reshape(n_new, self.d - 1, 3, 3)

        new_chi, Z = self._chi_update_batch(parents)

        total_Z = Z.sum()
        if total_Z <= 0 or not np.isfinite(total_Z):
            self.no_update = True
            return

        weights = np.power(Z, self.m_parisi)
        w_sum = weights.sum()
        if w_sum <= 0 or not np.isfinite(w_sum):
            self.no_update = True
            return
        probs = weights / w_sum

        resample_idx = self.rng.choice(n_new, size=n_new, p=probs)
        new_chi = new_chi[resample_idx]

        if self.impose_symmetry:
            half = n_new // 2
            perm = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
            rotated = new_chi[:half][:, perm[1]][:, :, perm[1]]
            new_chi = np.concatenate([rotated, new_chi[half:]], axis=0)

        write_idx = self.rng.integers(0, self.M, size=n_new)
        self.population[write_idx] = new_chi
        self.no_update = False


    def update_observables(self, num_samples=None):
        if num_samples is None:
            num_samples = self.num_samples
        d = self.d

        idx = self.rng.integers(0, self.M, size=num_samples * d)
        parents_n = self.population[idx].reshape(num_samples, d, 3, 3)

        Zn = np.zeros(num_samples)
        H = self.H
        for x in range(3):
            f1, f2, f3 = self._fmap[x]
            a = parents_n[:, :, f1, x]
            b = parents_n[:, :, f2, x]
            c = parents_n[:, :, f3, x]

            dp = np.zeros((num_samples, d + 1))
            dp[:, 0] = 1.0
            for l in range(d):
                bc_l = b[:, l] + c[:, l]
                a_l = a[:, l]
                new_dp = dp * bc_l[:, None]
                new_dp[:, 1:] += dp[:, :-1] * a_l[:, None]
                dp = new_dp

            if H <= d:
                Zn += dp[:, H:d + 1].sum(axis=1)

        idx = self.rng.integers(0, self.M, size=num_samples * 2)
        parents_e = self.population[idx].reshape(num_samples, 2, 3, 3)
        Ze = np.zeros(num_samples)
        for x in range(3):
            for y in range(3):
                factor = np.exp((self.mu[x] + self.mu[y]) / d)
                Ze += factor * parents_e[:, 0, x, y] * parents_e[:, 1, y, x]

        m = self.m_parisi

        pow_n = np.power(Zn, m)
        pow_e = np.power(Ze, m)

        Zn_mean = pow_n.mean()
        Ze_mean = pow_e.mean()
        Zn_logmean = (pow_n * np.where(Zn > 0, np.log(np.where(Zn > 0, Zn, 1)), 0)).mean()
        Ze_logmean = (pow_e * np.where(Ze > 0, np.log(np.where(Ze > 0, Ze, 1)), 0)).mean()

        if Zn_mean <= 0 or Ze_mean <= 0:
            self.psi = self.phi = self.complexity = np.nan
            self.rho = np.full(3, np.nan)
            return

        self.psi = np.log(Zn_mean) - (d / 2.0) * np.log(Ze_mean)
        self.phi = Zn_logmean / Zn_mean - (d / 2.0) * Ze_logmean / Ze_mean
        self.complexity = self.psi - m * self.phi

        rho = np.zeros(3)
        for a_idx in range(3):
            num = np.zeros(num_samples)
            for x in range(3):
                for y in range(3):
                    weight = (1 if x == a_idx else 0) + (1 if y == a_idx else 0)
                    if weight == 0:
                        continue
                    factor = np.exp((self.mu[x] + self.mu[y]) / d)
                    num += 0.5 * weight * factor * parents_e[:, 0, x, y] * parents_e[:, 1, y, x]
            rho[a_idx] = (pow_e * num / np.where(Ze > 0, Ze, 1)).mean() / Ze_mean
        self.rho = rho

    def diff(self, n_bins=200):
        total = 0.0
        for i in range(3):
            for j in range(3):
                h_old, _ = np.histogram(self.old_population[:, i, j],
                                        bins=n_bins, range=(0, 1), density=True)
                h_new, _ = np.histogram(self.population[:, i, j],
                                        bins=n_bins, range=(0, 1), density=True)
                total += np.abs(h_new - h_old).sum()
        return total


    def run(self, max_iter=None, tol=None, check_convergence=True,
            convergence_check_interval=None, sampling_threshold=None,
            sampling_interval=None, reset_population=False, verbose=0,
            auto_wandb_init=True):
        if reset_population:
            self.reset_population()
        if max_iter is None: max_iter = self.max_iter
        if tol is None: tol = self.tol
        if convergence_check_interval is None: convergence_check_interval = self.convergence_check_interval
        if sampling_threshold is None: sampling_threshold = self.sampling_threshold
        if sampling_interval is None: sampling_interval = self.sampling_interval

        if self.use_wandb and auto_wandb_init and self._wandb_run is None:
            self.wandb_init()

        psi_l, phi_l, sigma_l, rho_l = [], [], [], []
        t0 = time.time()
        final_diff = None
        converged = False
        it = 0

        for it in range(max_iter):
            self.step()
            self._global_step += 1

            diff_val = None
            if check_convergence and it % convergence_check_interval == 0 and it > 0:
                diff_val = self.diff()
                final_diff = diff_val
                if verbose >= 2:
                    print(f"  iter {it}: diff = {diff_val:.4f}")
                if diff_val < tol:
                    if verbose >= 1:
                        print(f"Converged at iter {it} (diff={diff_val:.4f} < tol={tol})")
                    converged = True

            sampled_now = False
            if it >= sampling_threshold and it % sampling_interval == 0:
                self.update_observables()
                psi_l.append(self.psi)
                phi_l.append(self.phi)
                sigma_l.append(self.complexity)
                rho_l.append(self.rho)
                sampled_now = True

            if self._wandb_run is not None:
                should_log = (it % self.log_every == 0) or sampled_now or (diff_val is not None)
                should_log_hist = (it % self.log_histograms_every == 0)
                if should_log or should_log_hist:
                    self._log_step_metrics(
                        it=it,
                        t_elapsed=time.time() - t0,
                        diff_val=diff_val,
                        include_observables=sampled_now,
                        include_histograms=should_log_hist,
                    )

            if converged:
                break

        self.update_observables()
        psi_l.append(self.psi)
        phi_l.append(self.phi)
        sigma_l.append(self.complexity)
        rho_l.append(self.rho)

        psi_l = np.array(psi_l)
        phi_l = np.array(phi_l)
        sigma_l = np.array(sigma_l)
        rho_l = np.array(rho_l)

        self.psi_mean, self.psi_std = np.nanmean(psi_l), np.nanstd(psi_l)
        self.phi_mean, self.phi_std = np.nanmean(phi_l), np.nanstd(phi_l)
        self.complexity_mean = np.nanmean(sigma_l)
        self.complexity_std = np.nanstd(sigma_l)
        self.rho_mean = np.nanmean(rho_l, axis=0)
        self.rho_std = np.nanstd(rho_l, axis=0)

        if verbose >= 1:
            print(f"Final: Ψ={self.psi_mean:.5f}, φ={self.phi_mean:.5f}, "
                  f"Σ={self.complexity_mean:.5f}, ρ={self.rho_mean}")

        # wandb summary
        if self._wandb_run is not None:
            wandb.summary['converged'] = bool(converged)
            wandb.summary['final_diff'] = (float(final_diff)
                                           if final_diff is not None else None)
            wandb.summary['total_iters'] = int(it + 1)
            wandb.summary['total_time_sec'] = float(time.time() - t0)
            wandb.summary['psi_mean'] = float(self.psi_mean)
            wandb.summary['psi_std'] = float(self.psi_std)
            wandb.summary['phi_mean'] = float(self.phi_mean)
            wandb.summary['phi_std'] = float(self.phi_std)
            wandb.summary['complexity_mean'] = float(self.complexity_mean)
            wandb.summary['complexity_std'] = float(self.complexity_std)
            for a in range(3):
                wandb.summary[f'rho_mean_{a}'] = float(self.rho_mean[a])
                wandb.summary[f'rho_std_{a}'] = float(self.rho_std[a])
                wandb.summary[f'mu_final_{a}'] = float(self.mu[a])


    def compute_complexity_curves(self, m_parisi_list=None, check_convergence=False,
                                  verbose=0, one_wandb_run_per_m=False):
        if m_parisi_list is not None:
            self.m_parisi_list = np.asarray(m_parisi_list)
        n = len(self.m_parisi_list)
        self.psi_list = np.zeros(n)
        self.psi_list_std = np.zeros(n)
        self.phi_list = np.zeros(n)
        self.phi_list_std = np.zeros(n)
        self.complexity_list = np.zeros(n)
        self.complexity_list_std = np.zeros(n)
        self.rho_list = np.zeros((n, 3))
        self.rho_list_std = np.zeros((n, 3))

        sweep_run = None
        if self.use_wandb and not one_wandb_run_per_m:
            sweep_run = self.wandb_init(
                name=(self.wandb_name or f"sweep_d{self.d}_H{self.H}"),
                group=self.wandb_group or f"sweep_d{self.d}_H{self.H}",
            )

        for i, mp in enumerate(self.m_parisi_list):
            if verbose >= 1:
                print(f"========= m_parisi = {mp} =========")
            self.m_parisi = mp

            if self.use_wandb and one_wandb_run_per_m:
                self.wandb_init(
                    name=f"d{self.d}_H{self.H}_m{mp:.4f}",
                    group=self.wandb_group or f"sweep_d{self.d}_H{self.H}",
                    reinit=True,
                )

            self.run(check_convergence=check_convergence,
                     reset_population=True, verbose=verbose,
                     auto_wandb_init=False)

            self.psi_list[i] = self.psi_mean
            self.psi_list_std[i] = self.psi_std
            self.phi_list[i] = self.phi_mean
            self.phi_list_std[i] = self.phi_std
            self.complexity_list[i] = self.complexity_mean
            self.complexity_list_std[i] = self.complexity_std
            self.rho_list[i] = self.rho_mean
            self.rho_list_std[i] = self.rho_std

            if sweep_run is not None:
                wandb.log({
                    'sweep/m_parisi': float(mp),
                    'sweep/psi_mean': float(self.psi_mean),
                    'sweep/psi_std': float(self.psi_std),
                    'sweep/phi_mean': float(self.phi_mean),
                    'sweep/phi_std': float(self.phi_std),
                    'sweep/complexity_mean': float(self.complexity_mean),
                    'sweep/complexity_std': float(self.complexity_std),
                    'sweep/rho_0_mean': float(self.rho_mean[0]),
                    'sweep/rho_1_mean': float(self.rho_mean[1]),
                    'sweep/rho_2_mean': float(self.rho_mean[2]),
                }, commit=True)

            if self.use_wandb and one_wandb_run_per_m:
                self.wandb_finish()

        # fit Σ(φ)
        def _fit_func(x, a, b, c, d):
            return a + b * np.power(2, x) + c * np.power(3, x)

        if np.nanmax(self.complexity_list) < 0:
            if verbose >= 1:
                print("No Σ=0 crossing: problem likely UNSAT at these m values.")
            self.phi_s = np.nanmax(self.phi_list)
            self.phi_d = None
        else:
            imax = int(np.nanargmax(self.complexity_list))
            try:
                self.fitting_param_phi, _ = optimize.curve_fit(
                    _fit_func, self.phi_list[imax:], self.complexity_list[imax:],
                    method='lm')
                phi_samples = np.linspace(np.nanmin(self.phi_list),
                                          np.nanmax(self.phi_list), 100000)
                self.phi_s = phi_samples[np.argmin(
                    np.abs(_fit_func(phi_samples, *self.fitting_param_phi)))]
                self.phi_d = self.phi_list[imax]
            except Exception as e:
                if verbose >= 1:
                    print(f"Fit failed: {e}")
                self.phi_s = self.phi_d = None

        if sweep_run is not None:
            if self.phi_s is not None:
                wandb.summary['phi_s'] = float(self.phi_s)
            if self.phi_d is not None:
                wandb.summary['phi_d'] = float(self.phi_d)
            try:
                table = wandb.Table(
                    columns=['m_parisi', 'phi', 'phi_std', 'sigma', 'sigma_std', 'psi'],
                    data=[[float(self.m_parisi_list[i]),
                           float(self.phi_list[i]), float(self.phi_list_std[i]),
                           float(self.complexity_list[i]), float(self.complexity_list_std[i]),
                           float(self.psi_list[i])] for i in range(n)]
                )
                wandb.log({
                    'complexity_curve': wandb.plot.line(
                        table, x='phi', y='sigma', title='Σ vs φ'),
                    'sigma_vs_m': wandb.plot.line(
                        table, x='m_parisi', y='sigma', title='Σ vs m_parisi'),
                    'phi_vs_m': wandb.plot.line(
                        table, x='m_parisi', y='phi', title='φ vs m_parisi'),
                })
            except Exception:
                pass
            self.wandb_finish()


    def draw_sigma_phi(self, errorbars=False, title=None):
        if title is None:
            title = f"d={self.d}, H={self.H}, μ={self.mu}"
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.axhline(0, linestyle='--', color='k')
        if self.phi_s is not None:
            ax.axvline(self.phi_s, linestyle='--', color='grey', label=r'$\phi_s$')
        if errorbars:
            ax.errorbar(self.phi_list, self.complexity_list,
                        xerr=self.phi_list_std, yerr=self.complexity_list_std,
                        fmt='o', capsize=3)
        else:
            ax.plot(self.phi_list, self.complexity_list, 'o')
        if self.fitting_param_phi is not None:
            xs = np.linspace(min(self.phi_list), max(self.phi_list), 500)
            def _fit(x, a, b, c, d): return a + b*np.power(2, x) + c*np.power(3, x)
            ax.plot(xs, _fit(xs, *self.fitting_param_phi), '-', label='fit')
        ax.set_xlabel(r'$\phi_\mathrm{int}$')
        ax.set_ylabel(r'$\Sigma$')
        ax.legend()
        return fig, ax

    def draw_population(self, n_bins=50, title=None):
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        fig.suptitle(title or f"d={self.d}, H={self.H}")
        for i in range(3):
            for j in range(3):
                axes[i, j].hist(self.population[:, i, j], bins=n_bins,
                                range=(0, 1), density=True)
                axes[i, j].set_xlabel(rf'$\chi_{{e_{i+1}, e_{j+1}}}$')
        plt.tight_layout()
        return fig

    def _default_folder(self):
        return f"results_experiments/3ass_popdynamics/d{self.d}_H{self.H}m_parisi={self.m_parisi}"

    def save_parameters(self, path):
        params = {
            'd': self.d, 'H': self.H,
            'm_parisi': float(self.m_parisi),
            'mu': self.mu.tolist(),
            'm_target': None if self.m_target is None else self.m_target.tolist(),
            'M': self.M, 'num_samples': self.num_samples,
            'damping': self.damping, 'init_type': self.init_type,
            'init_chi_bp': None if self.init_chi_bp is None else np.asarray(self.init_chi_bp).tolist(),
            'init_noise': self.init_noise,
            'mu_update': self.mu_update, 'mu_solver_loss': self.mu_solver_loss,
            'impose_symmetry': self.impose_symmetry,
            'max_iter': self.max_iter, 'tol': self.tol,
            'convergence_check_interval': self.convergence_check_interval,
            'sampling_threshold': self.sampling_threshold,
            'sampling_interval': self.sampling_interval,
            'm_parisi_list': self.m_parisi_list.tolist(),
            'seed': self.seed,
        }
        with open(path, 'w') as f:
            json.dump(params, f, indent=2)

    def save_observables(self, path):
        obs = {
            'psi': self.psi, 'phi': self.phi, 'complexity': self.complexity,
            'rho': None if self.rho is None else np.asarray(self.rho).tolist(),
            'psi_mean': self.psi_mean, 'psi_std': self.psi_std,
            'phi_mean': self.phi_mean, 'phi_std': self.phi_std,
            'complexity_mean': self.complexity_mean,
            'complexity_std': self.complexity_std,
            'rho_mean': None if self.rho_mean is None else np.asarray(self.rho_mean).tolist(),
            'rho_std': None if self.rho_std is None else np.asarray(self.rho_std).tolist(),
        }
        with open(path, 'w') as f:
            json.dump(obs, f, indent=2)

    def save_complexity_curves(self, path_csv, path_json):
        if self.phi_list is None:
            return
        with open(path_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['m_parisi',
                        'psi_mean', 'psi_std',
                        'phi_mean', 'phi_std',
                        'complexity_mean', 'complexity_std',
                        'rho1_mean', 'rho1_std',
                        'rho2_mean', 'rho2_std',
                        'rho3_mean', 'rho3_std'])
            for i in range(len(self.m_parisi_list)):
                w.writerow([
                    self.m_parisi_list[i],
                    self.psi_list[i], self.psi_list_std[i],
                    self.phi_list[i], self.phi_list_std[i],
                    self.complexity_list[i], self.complexity_list_std[i],
                    self.rho_list[i, 0], self.rho_list_std[i, 0],
                    self.rho_list[i, 1], self.rho_list_std[i, 1],
                    self.rho_list[i, 2], self.rho_list_std[i, 2],
                ])
        params = {
            'fitting_param_phi': None if self.fitting_param_phi is None else self.fitting_param_phi.tolist(),
            'phi_s': self.phi_s, 'phi_d': self.phi_d,
        }
        with open(path_json, 'w') as f:
            json.dump(params, f, indent=2)

    def save_population(self, path):
        np.save(path, self.population)

    def save(self, folder=None, save_pop=False):
        if folder is None:
            folder = self._default_folder()
        os.makedirs(folder, exist_ok=True)
        self.save_parameters(os.path.join(folder, 'parameters.json'))
        self.save_observables(os.path.join(folder, 'observables.json'))
        self.save_complexity_curves(
            os.path.join(folder, 'complexity_curves.csv'),
            os.path.join(folder, 'complexity_curves.json'),
        )
        if save_pop:
            self.save_population(os.path.join(folder, 'population.npy'))


if __name__ == '__main__':
    pd = population_dynamics_3ass(
        d=5, H=3,
        m_parisi=1.0,
        mu=np.zeros(3),
        M=2000,
        num_samples=5000,
        damping=0.8,
        init_type='hard_fields',
        impose_symmetry=False,
        max_iter=500,
        tol=5.0,
        sampling_threshold=200,
        sampling_interval=20,
        seed=0,
        use_wandb=False,
    )
    pd.run(verbose=1)
    pd.compute_complexity_curves(verbose=1)
    pd.save()  
    print(pd)

    # pd_wandb = population_dynamics_3ass(
    #     d=5, H=3, m_parisi=1.0, M=5000,
    #     m_target=np.array([1/3, 1/3, 1/3]), mu_update='target_m',
    #     init_type='almost_unif',
    #     max_iter=2000, tol=5.0,
    #     sampling_threshold=500, sampling_interval=50,
    #     seed=0,
    #     use_wandb=True,
    #     wandb_project='pop_dyn_3ass',
    #     wandb_group='d5_H3_sweep',
    #     wandb_name='balanced_m1',
    #     log_every=50,
    #     log_histograms_every=500,
    # )
    # pd_wandb.run(verbose=1)
    # pd_wandb.wandb_finish()