"""Population dynamics for H-(dis)assortative balanced K-partitions on
random d-regular graphs."""


from __future__ import annotations

import os
import time
import math
from typing import Any, Dict, List, Literal, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ImportError as exc:
    raise ImportError("population_dynamics_optimized.py requires PyTorch.") from exc

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


ProblemType = Literal["assortative", "disassortative"]
InitType = Literal[
    "rs_exact",
    "rs_perturb",
    "hard_field",
    "almost_uniform",
    "gaussian",
    "manual_population",
    "manual_chi",
]


class PopDyn:
    def __init__(
        self,

        K: int,
        d: int,
        H: int,
        problem_type: ProblemType = "assortative",

        mparisi: float = 1.0,

        M: int = 10_000,
        damping: float = 0.8,
        max_iter: int = 10_000,
        tol: float = 1e-8,
        convergence_check_every: int = 200,
        track_diff: bool = True,
        diff_n_bins: int = 200,
        eps: float = 1e-300,

        num_samples: Optional[int] = None,
        observable_upsampling_factor: int = 10,
        observable_batch_size: int = 20_000,
        min_observable_samples: int = 20,
        sampling_start_iter: int = 8_000,
        sampling_interval: int = 50,
        require_convergence_for_sampling: bool = True,
        min_valid_fraction: float = 0.995,  # validity of Monte Carlo samples used for observables
        max_discarded_message_fraction: float = 0.05,  # health of the actual pop-dyn update
        implosion_check_start_iter: int = 500,
        min_valid_candidates_for_update: int = 100,

        init_type: InitType = "hard_field",
        init_noise: float = 1e-6,
        chi_RS: Optional[np.ndarray] = None,
        init_population: Optional[np.ndarray] = None,
        manual_init_chi: Optional[np.ndarray] = None,

        impose_color_symmetry: bool = True,

        seed: Optional[int] = None,
        dtype: np.dtype = np.float64,
        device: Optional[str] = None,
        compile_core: bool = False,

        use_wandb: bool = False,
        wandb_project: str = "bp_pop_dyn",
        wandb_group: Optional[str] = None,
        wandb_name: Optional[str] = None,
        wandb_config_extra: Optional[Dict[str, Any]] = None,
        log_every: int = 50,
        wandb_log_plots: bool = True,
        save_final_plots: bool = True,
        final_plot_bins: int = 100,
        save_locally: bool = True,
        save_dir: str = "results/pop_dyn",
        run_name: Optional[str] = None,

        diagnostic_every: Optional[int] = None,
        diagnostic_hist_bins: int = 80,
        diagnostic_sample_size: int = 100_000,
        save_diagnostic_plots: bool = True,
    ) -> None:

        self.K = int(K)
        self.d = int(d)
        self.H = int(H)
        self.problem_type = problem_type

        self.target_density = np.full(self.K, 1.0 / self.K, dtype=dtype)
        self.mu = np.zeros(self.K, dtype=dtype)

        self.mparisi = float(mparisi)

        self.M = int(M)
        self.damping = float(damping)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.convergence_check_every = int(convergence_check_every)
        self.track_diff = bool(track_diff)
        self.diff_n_bins = int(diff_n_bins)
        self.eps = float(eps)
        self.dtype = np.dtype(dtype)
        self.torch_dtype = (
            torch.float32
            if self.dtype == np.dtype(np.float32)
            else torch.float64
        )
        self.torch_eps = max(self.eps, float(torch.finfo(self.torch_dtype).tiny))
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        self.compile_core = bool(compile_core)

        self.num_parents = self.d - 1
        self.num_updated = max(1, round((1.0 - self.damping) * self.M))
        self.num_parent_draws = self.num_updated * self.num_parents

        self.message_shape: Tuple[int, int] = (self.K, self.K)
        self.population_shape: Tuple[int, int, int] = (self.M, self.K, self.K)

        self.num_samples = None if num_samples is None else int(num_samples)
        self.observable_upsampling_factor = int(observable_upsampling_factor)
        self.observable_batch_size = int(observable_batch_size)

        if self.observable_batch_size <= 0:
            raise ValueError("observable_batch_size must be strictly positive.")
        self.min_observable_samples = int(min_observable_samples)
        self.sampling_start_iter = int(sampling_start_iter)
        self.sampling_interval = int(sampling_interval)
        self.require_convergence_for_sampling = bool(require_convergence_for_sampling)
        self.min_valid_fraction = float(min_valid_fraction)
        self.last_observables_valid = False
        self.max_discarded_message_fraction = float(max_discarded_message_fraction)
        self.consecutive_no_valid_updates = 0
        self.successful_updates = 0
        self.min_valid_candidates_for_update = int(min_valid_candidates_for_update)

        if not 0.0 <= self.max_discarded_message_fraction <= 1.0:
            raise ValueError("max_discarded_message_fraction must be between 0 and 1.")

        self.init_type = init_type
        self.init_noise = float(init_noise)

        valid_init_types = {
            "rs_exact",
            "rs_perturb",
            "hard_field",
            "almost_uniform",
            "gaussian",
            "manual_population",
            "manual_chi",
        }

        if self.init_type not in valid_init_types:
            raise ValueError(
                f"Unknown init_type={self.init_type!r}. "
                f"Valid options are {sorted(valid_init_types)}."
            )

        self.chi_RS = None if chi_RS is None else self._normalize_message(chi_RS)
        self.init_population = init_population
        self.manual_init_chi = (None if manual_init_chi is None else self._normalize_message(manual_init_chi))

        if self.init_type in ("rs_exact", "rs_perturb") and self.chi_RS is None:
            raise ValueError(f"init_type={self.init_type!r} requires chi_RS to be provided.")
        if self.init_type == "manual_population" and self.init_population is None:
            raise ValueError("init_type='manual_population' requires init_population.")
        if self.init_type == "manual_chi" and self.manual_init_chi is None:
            raise ValueError("init_type='manual_chi' requires manual_init_chi.")

        self.impose_color_symmetry = bool(impose_color_symmetry)

        self.color_labels = np.arange(self.K)

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.torch_generator = torch.Generator(device=self.device)
        if seed is not None:
            self.torch_generator.manual_seed(int(seed))

        if use_wandb and not WANDB_AVAILABLE:
            raise ImportError(
                "use_wandb=True, but wandb is not installed in this Python environment. "
                "Install it with: python -m pip install wandb"
            )
        self.use_wandb = bool(use_wandb)
        self.wandb_project = wandb_project
        self.wandb_group = wandb_group
        self.wandb_name = wandb_name
        self.wandb_config_extra = wandb_config_extra or {}
        self.log_every = int(log_every)
        self.wandb_log_plots = bool(wandb_log_plots)
        self.save_final_plots = bool(save_final_plots)
        self.final_plot_bins = int(final_plot_bins)

        self.save_locally = bool(save_locally)
        self.save_dir = save_dir

        if run_name is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.run_name = (
                f"PD_{self.problem_type[:3]}"
                f"_K{self.K}_d{self.d}_H{self.H}"
                f"_m{self.mparisi:.3f}"
                f"_{self.init_type}_{timestamp}"
            )
        else:
            self.run_name = run_name

        self.diagnostic_every = (None if diagnostic_every is None else int(diagnostic_every))
        self.diagnostic_hist_bins = int(diagnostic_hist_bins)
        self.diagnostic_sample_size = int(diagnostic_sample_size)
        self.save_diagnostic_plots = bool(save_diagnostic_plots)
        self.population_diagnostic_history: List[Dict[str, Any]] = []

        self.observable_iterations: List[int] = []

        if self.K <= 0:
            raise ValueError("K must be positive.")
        if self.d <= 0:
            raise ValueError("d must be positive.")
        if self.M <= 0:
            raise ValueError("M must be positive.")
        if not 0.0 <= self.damping < 1.0:
            raise ValueError("damping must satisfy 0 <= damping < 1.")
        if self.num_updated > self.M:
            raise ValueError("num_updated cannot exceed M.")
        if self.diagnostic_every is not None and self.diagnostic_every <= 0:
            raise ValueError("diagnostic_every must be positive or None.")
        if self.diagnostic_hist_bins <= 0:
            raise ValueError("diagnostic_hist_bins must be positive.")
        if self.diagnostic_sample_size <= 0:
            raise ValueError("diagnostic_sample_size must be positive.")

        initial_population = self._initialize_population()

        print(
            "[initialization] "
            f"init_type={self.init_type}, "
            f"shape={initial_population.shape}, "
            f"numpy_dtype={initial_population.dtype}, "
            f"torch_dtype={self.torch_dtype}, "
            f"device={self.device}",
            flush=True,
        )

        numpy_finite = np.isfinite(initial_population)

        if not np.all(numpy_finite):
            nan_count = int(np.isnan(initial_population).sum())
            posinf_count = int(np.isposinf(initial_population).sum())
            neginf_count = int(np.isneginf(initial_population).sum())

            raise RuntimeError(
                "Initial NumPy population contains non-finite values: "
                f"init_type={self.init_type}, "
                f"nan={nan_count}, "
                f"+inf={posinf_count}, "
                f"-inf={neginf_count}"
            )

        if self.init_type == "hard_field":
            valid_components = (
                (initial_population == 0.0)
                | (initial_population == 1.0)
            )

            if not np.all(valid_components):
                invalid_count = int((~valid_components).sum())

                raise RuntimeError(
                    "Hard-field initialization contains values other than 0 and 1: "
                    f"invalid_components={invalid_count}"
                )

            numpy_mass = initial_population.sum(axis=(1, 2))

            if not np.all(numpy_mass == 1.0):
                bad_messages = int((numpy_mass != 1.0).sum())

                raise RuntimeError(
                    "Hard-field initialization contains non-one-hot messages: "
                    f"bad_messages={bad_messages}, "
                    f"mass_min={numpy_mass.min()}, "
                    f"mass_max={numpy_mass.max()}"
                )

        # Force an owned contiguous array before transferring it.
        initial_population = np.ascontiguousarray(initial_population).copy()

        self.population = torch.from_numpy(initial_population).to(
            device=self.device,
            dtype=self.torch_dtype,
        )

        self.old_population = self.population.clone()

        torch_finite = torch.isfinite(self.population)

        if not bool(torch_finite.all().item()):
            nan_count = int(torch.isnan(self.population).sum().item())
            posinf_count = int(torch.isposinf(self.population).sum().item())
            neginf_count = int(torch.isneginf(self.population).sum().item())

            raise RuntimeError(
                "Population became non-finite after conversion to PyTorch: "
                f"init_type={self.init_type}, "
                f"device={self.device}, "
                f"nan={nan_count}, "
                f"+inf={posinf_count}, "
                f"-inf={neginf_count}"
            )

        initial_mass = self.population.sum(dim=(1, 2))
        initial_max_error = torch.abs(initial_mass - 1.0).max()

        if float(initial_max_error.item()) > 1e-12:
            raise RuntimeError(
                "Initial population is not normalized: "
                f"mass_min={initial_mass.min().item():.16g}, "
                f"mass_max={initial_mass.max().item():.16g}, "
                f"max_error={initial_max_error.item():.3e}"
            )

        if self.population.shape != (self.M, self.K, self.K):
            raise RuntimeError(
                f"Incorrect initialized population shape: "
                f"got {tuple(self.population.shape)}, "
                f"expected {(self.M, self.K, self.K)}"
            )

        self._diag_idx = torch.arange(self.K, device=self.device)
        self._r_parent = torch.arange(self.num_parents + 1, device=self.device)
        self._r_node = torch.arange(self.d + 1, device=self.device)
        self._mask_diag = self._constraint_mask_tensor(self._r_parent + 1)
        self._mask_offdiag = self._constraint_mask_tensor(self._r_parent)
        self._mask_node = self._constraint_mask_tensor(self._r_node)
        self._field_message = torch.exp(-(torch.as_tensor(self.mu, dtype=self.torch_dtype, device=self.device)[:, None]
            + torch.as_tensor(self.mu, dtype=self.torch_dtype, device=self.device)[None, :]) / self.d)
        self._field_node = torch.exp(-torch.as_tensor(self.mu, dtype=self.torch_dtype, device=self.device))
        self._has_message_field = bool(np.any(self.mu != 0.0))
        self._has_node_field = self._has_message_field

        self.no_update = False
        self.imploded = False
        self.implosion_iteration: Optional[int] = None
        self.implosion_check_start_iter = int(implosion_check_start_iter)

        self.last_diff: Optional[float] = None
        self.iteration = 0

        self.psi: Optional[float] = None
        self.phi: Optional[float] = None
        self.complexity: Optional[float] = None
        self.rho: Optional[np.ndarray] = None
        self.s: Optional[float] = None

        self.psi_samples: List[float] = []
        self.phi_samples: List[float] = []
        self.complexity_samples: List[float] = []
        self.rho_samples: List[np.ndarray] = []
        self.s_samples: List[float] = []

        self.psi_mean: Optional[float] = None
        self.phi_mean: Optional[float] = None
        self.complexity_mean: Optional[float] = None
        self.rho_mean: Optional[np.ndarray] = None
        self.s_mean: Optional[float] = None

        self.psi_std: Optional[float] = None
        self.phi_std: Optional[float] = None
        self.complexity_std: Optional[float] = None
        self.rho_std: Optional[np.ndarray] = None
        self.s_std: Optional[float] = None

        self.m_list: Optional[np.ndarray] = None

        self.psi_list: Optional[np.ndarray] = None
        self.psi_list_std: Optional[np.ndarray] = None

        self.phi_list: Optional[np.ndarray] = None
        self.phi_list_std: Optional[np.ndarray] = None

        self.complexity_list: Optional[np.ndarray] = None
        self.complexity_list_std: Optional[np.ndarray] = None

        self.rho_list: Optional[np.ndarray] = None
        self.rho_list_std: Optional[np.ndarray] = None

        self.s_list: Optional[np.ndarray] = None
        self.s_list_std: Optional[np.ndarray] = None

        self.phi_d: Optional[float] = None
        self.phi_s: Optional[float] = None
        self.rho_d: Optional[np.ndarray] = None
        self.rho_s: Optional[np.ndarray] = None

        self.fitting_param_phi: Optional[np.ndarray] = None
        self.fitting_param_rho: Optional[np.ndarray] = None

        self.phase: Optional[str] = None
        self.diagnostics: Dict[str, Any] = {}

        self.wandb_run = None
        if self.use_wandb:
            self._init_wandb()


    def __repr__(self) -> str:
        lines = [
            "PopDyn",
            f"  problem        : {self.problem_type}",
            f"  K, d, H        : {self.K}, {self.d}, {self.H}",
            f"  m              : {self.mparisi:.6g}",
            f"  M              : {self.M}",
            f"  device         : {self.device}",
            f"  dtype          : {self.torch_dtype}",
            f"  damping        : {self.damping}",
            f"  init_type      : {self.init_type}",
            f"  color symmetry : {self.impose_color_symmetry}",
            f"  iteration      : {self.iteration}",
            f"  obs upsampling : {self.observable_upsampling_factor}x",
            f"  obs samples    : {len(self.psi_samples)}",
        ]

        if self.phase is not None:
            lines.append(f"  phase          : {self.phase}")

        if self.last_diff is not None:
            lines.append(f"  last diff      : {self.last_diff:.6g}")

        if self.psi_mean is not None:
            lines.extend([
                "",
                "Observables",
                f"  Ψ mean         : {self.psi_mean:.8g} ± {self.psi_std:.3g}",
                f"  φ mean         : {self.phi_mean:.8g} ± {self.phi_std:.3g}",
                f"  Σ mean         : {self.complexity_mean:.8g} ± {self.complexity_std:.3g}",
                f"  s mean         : {self.s_mean:.8g} ± {self.s_std:.3g}",
            ])

            if self.rho_mean is not None:
                rho_str = np.array2string(self.rho_mean, precision=5, suppress_small=True,)
                rho_std_str = np.array2string(self.rho_std, precision=2, suppress_small=True,)
                lines.append(f"  rho mean       : {rho_str}")
                lines.append(f"  rho std        : {rho_std_str}")

        elif self.psi is not None:
            lines.extend([
                "",
                "Current observables",
                f"  Ψ              : {self.psi:.8g}",
                f"  φ              : {self.phi:.8g}",
                f"  Σ              : {self.complexity:.8g}",
                f"  s              : {self.s:.8g}",
            ])

            if self.rho is not None:
                rho_str = np.array2string(self.rho, precision=5, suppress_small=True,)
                lines.append(f"  rho            : {rho_str}")

        return "\n".join(lines)

    # -----------------------------------------------------------
    # HELPER FUNCTIONS
    # -----------------------------------------------------------

    def _normalize_message(self, chi: np.ndarray) -> np.ndarray:
        chi = np.asarray(chi, dtype=self.dtype)
        chi = np.where(np.isfinite(chi), chi, 0.0)
        chi = np.maximum(chi, 0.0)

        mass = float(chi.sum())
        if not np.isfinite(mass) or mass <= self.eps:
            return np.full(self.message_shape, 1.0 / (self.K * self.K), dtype=self.dtype,)

        return chi / mass


    @torch.inference_mode()
    def _normalize_messages_torch(
        self,
        messages: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if messages.ndim != 3 or messages.shape[1:] != self.message_shape:
            raise ValueError(
                "Expected messages with shape "
                f"(B, {self.K}, {self.K}), got {tuple(messages.shape)}."
            )

        finite_messages = torch.isfinite(messages).all(dim=(1, 2))
        cleaned = torch.where(torch.isfinite(messages), messages,torch.zeros_like(messages),).clamp_min(0.0)

        mass = cleaned.sum(dim=(1, 2), keepdim=True)
        scalar_mass = mass[:, 0, 0]
        bad = (~finite_messages | ~torch.isfinite(scalar_mass) | (scalar_mass <= self.torch_eps))

        normalized = cleaned / mass.clamp_min(self.torch_eps)
        uniform = torch.full_like(normalized, 1.0 / (self.K * self.K),)
        normalized = torch.where(bad[:, None, None], uniform, normalized,)

        return normalized, bad


    def _normalize_population(self, pop: np.ndarray) -> np.ndarray:
        pop = np.asarray(pop, dtype=self.dtype)

        if pop.shape != self.population_shape:
            raise ValueError(
                f"Population must have shape {self.population_shape}, "
                f"got {pop.shape}."
            )

        pop = np.where(np.isfinite(pop), pop, 0.0)
        pop = np.maximum(pop, 0.0)

        mass = pop.sum(axis=(1, 2), keepdims=True)
        bad = (~np.isfinite(mass[:, 0, 0]) | (mass[:, 0, 0] <= self.eps))

        if np.any(bad):
            pop[bad] = 1.0 / (self.K * self.K)
            mass = pop.sum(axis=(1, 2), keepdims=True)

        return pop / mass

    # -----------------------------------------------------------
    # INITIALIZE POPULATION
    # -----------------------------------------------------------

    def _initialize_population(self) -> np.ndarray:
        if self.init_type == "manual_population":
            return self._normalize_population(self.init_population)

        if self.init_type == "manual_chi":
            pop = np.repeat(self.manual_init_chi[None, :, :], self.M, axis=0)
            return self._normalize_population(pop)

        if self.init_type == "rs_exact":
            pop = np.repeat(self.chi_RS[None, :, :], self.M, axis=0)
            return self._normalize_population(pop)

        if self.init_type == "rs_perturb":
            pop = np.repeat(self.chi_RS[None, :, :], self.M, axis=0)
            noise = self.rng.normal(loc=0.0, scale=self.init_noise, size=self.population_shape,)
            pop = pop + noise
            return self._normalize_population(pop)
        
        if self.init_type == "hard_field":
            pop = np.zeros(self.population_shape, dtype=self.dtype)

            flat_indices = self.rng.integers(low=0, high=self.K * self.K, size=self.M,)

            rows = flat_indices // self.K
            cols = flat_indices % self.K

            pop[np.arange(self.M), rows, cols] = 1.0
            return pop

        if self.init_type == "almost_uniform":
            pop = np.full(self.population_shape, 1.0 / (self.K * self.K), dtype=self.dtype,)
            noise = self.rng.normal(loc=0.0, scale=self.init_noise, size=self.population_shape,)
            pop = pop + noise
            return self._normalize_population(pop)

        if self.init_type == "gaussian":
            pop = self.rng.normal(loc=1.0, scale=max(self.init_noise, 1e-3), size=self.population_shape,)
            return self._normalize_population(pop)
    
    #=============================================================================================================================
    # POPULATION DYNAMICS RUN
    #=============================================================================================================================

    def _constraint_mask_from_same_count(self, same_count):
        if self.problem_type == "assortative":
            return same_count >= self.H
        return same_count < self.H


    def _constraint_mask_tensor(self, same_count: torch.Tensor) -> torch.Tensor:
        if self.problem_type == "assortative":
            return same_count >= self.H
        return same_count < self.H


    def _constraint_satisfied(self, x: int, y: int, parent_colors: np.ndarray) -> bool:
        parent_colors = np.asarray(parent_colors)
        same_count = int(x == y) + np.count_nonzero(parent_colors == x)
        return bool(self._constraint_mask_from_same_count(same_count))


    def _same_count_coefficients_torch(self, parent_messages: torch.Tensor) -> torch.Tensor:
        B, P = parent_messages.shape[:2]
        if P == 0:
            return torch.ones((B, self.K, 1), dtype=self.torch_dtype, device=self.device)

        same = parent_messages[:, :, self._diag_idx, self._diag_idx]
        different = parent_messages.sum(dim=2) - same

        coeff = torch.ones((B, self.K, 1), dtype=self.torch_dtype, device=self.device)
        for ell in range(P):
            diff_l = different[:, ell, :, None]
            same_l = same[:, ell, :, None]
            next_coeff = torch.zeros((B, self.K, ell + 2), dtype=self.torch_dtype, device=self.device)
            next_coeff[:, :, :ell + 1].add_(coeff * diff_l)
            next_coeff[:, :, 1:].add_(coeff * same_l)
            coeff = next_coeff
        return coeff


    def _same_count_coefficients(self, parent_messages):
        tensor = torch.as_tensor(parent_messages, dtype=self.torch_dtype, device=self.device)
        return self._same_count_coefficients_torch(tensor)


    def _candidate_messages_torch(self, parent_messages: torch.Tensor):
        coeff = self._same_count_coefficients_torch(parent_messages)
        diag_values = (coeff * self._mask_diag).sum(dim=2)
        offdiag_values = (coeff * self._mask_offdiag).sum(dim=2)

        candidates = offdiag_values[:, :, None].expand(-1, -1, self.K).clone()
        candidates[:, self._diag_idx, self._diag_idx] = diag_values
        if self._has_message_field:
            candidates.mul_(self._field_message)
        return candidates, candidates.sum(dim=(1, 2))


    def _candidate_messages(self, parent_messages):
        tensor = torch.as_tensor(parent_messages, dtype=self.torch_dtype, device=self.device)
        return self._candidate_messages_torch(tensor)


    def _randomly_permute_colors_torch(
        self,
        messages: torch.Tensor,
    ) -> torch.Tensor:
        if messages.ndim != 3:
            raise ValueError(f"Expected messages of shape (B, K, K), got {messages.shape}")

        B, K1, K2 = messages.shape

        if K1 != self.K or K2 != self.K:
            raise ValueError(
                f"Expected trailing shape ({self.K}, {self.K}), "
                f"got ({K1}, {K2})"
            )

        # generate valid permutations on CPU
        random_values = self.rng.random((B, self.K))
        perms_np = np.argsort(random_values, axis=1).astype(np.int64)

        perms = torch.from_numpy(perms_np).to(device=self.device, dtype=torch.long,)

        permutation_matrices = torch.nn.functional.one_hot(perms, num_classes=self.K,).to(dtype=messages.dtype)

        return torch.bmm(torch.bmm(permutation_matrices, messages), permutation_matrices.transpose(1, 2),)


    def _randomly_permute_colors(self, messages):
        tensor = torch.as_tensor(messages, dtype=self.torch_dtype, device=self.device)
        return self._randomly_permute_colors_torch(tensor)
    

    @torch.inference_mode()
    def step(self, snapshot_old: bool = True) -> None:
        if snapshot_old:
            self.old_population.copy_(self.population)

        if self.num_parents == 0:
            parent_messages = torch.empty((self.num_updated, 0, self.K, self.K), dtype=self.torch_dtype,device=self.device,)
        else:
            parent_indices = torch.randint(low=0, high=self.M, size=(self.num_updated, self.num_parents), device=self.device, generator=self.torch_generator, dtype=torch.long,)
            parent_messages = torch.index_select(self.population, dim=0, index=parent_indices.reshape(-1),).reshape(self.num_updated, self.num_parents, self.K, self.K,)

        should_record = (self.diagnostic_every is not None and self.diagnostic_every > 0 and (self.iteration + 1) % self.diagnostic_every == 0)

        candidates, candidate_mass = self._candidate_messages_torch(parent_messages)
        finite_mass = torch.isfinite(candidate_mass)
        valid = (finite_mass & (candidate_mass > self.torch_eps))

        num_candidates = candidate_mass.numel()
        num_valid = int(valid.sum().item())
        num_discarded = num_candidates - num_valid

        discarded_fraction = (num_discarded / num_candidates if num_candidates > 0 else 1.0)

        self.diagnostics["last_candidate_count"] = num_candidates
        self.diagnostics["last_valid_candidate_count"] = num_valid
        self.diagnostics["last_discarded_candidate_count"] = num_discarded
        self.diagnostics["last_discarded_candidate_fraction"] = discarded_fraction
        self.diagnostics["total_discarded_candidates"] = (int(self.diagnostics.get("total_discarded_candidates", 0,)) + num_discarded)
        self.diagnostics["total_generated_candidates"] = (int(self.diagnostics.get("total_generated_candidates", 0,)) + num_candidates)

        if self.iteration >= self.implosion_check_start_iter and self.successful_updates >= 100 and discarded_fraction > self.max_discarded_message_fraction:
            self.imploded = True
            self.implosion_iteration = self.iteration + 1

            self.diagnostics["imploded"] = True
            self.diagnostics["implosion_iteration"] = self.implosion_iteration
            self.diagnostics["implosion_discarded_fraction"] = discarded_fraction
            self.diagnostics["implosion_reason"] = ("too_many_discarded_candidates")
            
            self.no_update = True
            self.iteration += 1

            if should_record:
                self._record_and_save_diagnostics(candidate_mass)

            return

        valid_indices = torch.nonzero(valid, as_tuple=False,).flatten()

        if valid_indices.numel() < self.min_valid_candidates_for_update:
            self.consecutive_no_valid_updates += 1

            self.diagnostics["last_no_valid_update"] = True
            self.diagnostics["insufficient_valid_candidates"] = True
            self.diagnostics["consecutive_insufficient_valid_updates"] = self.consecutive_no_valid_updates

            self.no_update = True
            self.iteration += 1

            if self.iteration >= self.implosion_check_start_iter and self.consecutive_no_valid_updates >= 5:
                self.imploded = True
                self.implosion_iteration = self.iteration

                self.diagnostics["imploded"] = True
                self.diagnostics["implosion_iteration"] = self.implosion_iteration
                self.diagnostics["implosion_reason"] = "persistent_no_valid_candidates"
                self.diagnostics["implosion_discarded_fraction"] = discarded_fraction


            if should_record:
                self._record_and_save_diagnostics(candidate_mass)

            return

        self.consecutive_no_valid_updates = 0
        self.diagnostics["last_no_valid_update"] = False
        self.diagnostics["consecutive_no_valid_updates"] = 0

        valid_candidates = torch.index_select(candidates, dim=0, index=valid_indices,)
        valid_mass = torch.index_select(candidate_mass, dim=0, index=valid_indices,)

        log_weights = self.mparisi * torch.log(valid_mass.clamp_min(self.torch_eps))
        log_weights -= log_weights.max()

        weights = torch.exp(log_weights)

        if not bool(torch.isfinite(weights).all().item()) or float(weights.sum().item()) <= self.torch_eps:
            self.imploded = True
            self.implosion_iteration = self.iteration + 1

            self.diagnostics["imploded"] = True
            self.diagnostics["implosion_iteration"] = (self.implosion_iteration)
            self.diagnostics["implosion_reason"] = ("invalid_resampling_weights")

            self.no_update = True
            self.iteration += 1
            return

        selected = torch.multinomial(weights, num_samples=self.num_updated, replacement=True, generator=self.torch_generator,)
        selected_candidates = torch.index_select(valid_candidates, dim=0, index=selected,)
        selected_mass = torch.index_select(valid_mass, dim=0, index=selected,)

        new_messages = selected_candidates / selected_mass[:, None, None]

        # checks
        new_message_mass = new_messages.sum(dim=(1, 2))
        max_normalization_error = torch.abs(
            new_message_mass - 1.0
        ).max()

        self.diagnostics["last_new_message_normalization_error"] = float(
            max_normalization_error.item()
        )

        if float(max_normalization_error.item()) > 1e-12:
            self.imploded = True
            self.implosion_iteration = self.iteration + 1
            self.diagnostics["imploded"] = True
            self.diagnostics["implosion_iteration"] = self.implosion_iteration
            self.diagnostics["implosion_reason"] = (
                "normalized_message_mass_error"
            )
            self.no_update = True
            self.iteration += 1
            return
        # checks

        if self.impose_color_symmetry:
            new_messages = self._randomly_permute_colors_torch(new_messages)

        population_size = int(self.population.shape[0])
        if population_size != self.M:
            raise RuntimeError(
                "Population-size mismatch: "
                f"shape[0]={population_size}, M={self.M}."
            )
        if new_messages.shape != (self.num_updated, self.K, self.K,):
            raise RuntimeError(
                "Invalid replacement batch shape: "
                f"got {tuple(new_messages.shape)}, expected "
                f"{(self.num_updated, self.K, self.K)}."
            )
        if not bool(torch.isfinite(new_messages).all().item()):
            self.imploded = True
            self.implosion_iteration = self.iteration + 1

            self.diagnostics["imploded"] = True
            self.diagnostics["implosion_iteration"] = (self.implosion_iteration)
            self.diagnostics["implosion_reason"] = ("nonfinite_normalized_messages")

            self.no_update = True
            self.iteration += 1
            return

        replace_indices = torch.randperm(population_size, device=self.device, generator=self.torch_generator, dtype=torch.long,)[:self.num_updated]

        self.population.index_copy_(dim=0, index=replace_indices, source=new_messages,)
        
        self.no_update = False
        self.iteration += 1
        self.successful_updates += 1

        if should_record:
            self._record_and_save_diagnostics(candidate_mass)


    def run(
        self,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
        check_convergence: bool = True,
        sample_observables: bool = True,
        num_samples: Optional[int] = None,
        reset_samples: bool = True,
        stable_checks_required: int = 5,
        verbose: int = 1,
        finish_wandb: bool = True,
    ) -> None:
        if max_iter is None:
            max_iter = self.max_iter
        if tol is None:
            tol = self.tol

        if reset_samples:
            self._reset_observable_samples()

        stable = False
        stable_checks = 0
        start_time = time.time()

        for _ in range(max_iter):
            will_check_diff = (check_convergence and self.track_diff and (self.iteration + 1) % self.convergence_check_every == 0)
            self.step(snapshot_old=will_check_diff)

            if self.imploded:
                if verbose >= 1:
                    print(
                        f"Population dynamics imploded at iter={self.iteration}. "
                        f"Reason: {self.diagnostics.get('implosion_reason', 'unknown')}. "
                        f"Discarded fraction: "
                        f"{self.diagnostics.get('implosion_discarded_fraction', float('nan')):.6g}"
                    )
                break

            if self.no_update:
                stable = False
                stable_checks = 0

                if verbose >= 2 and self.iteration % self.convergence_check_every == 0:
                    print(
                        f"iter={self.iteration} "
                        "no valid population update; "
                        f"consecutive_no_valid="
                        f"{self.consecutive_no_valid_updates}"
                    )

                if self.use_wandb and self.iteration % self.log_every == 0:
                    self._log_wandb()

                continue

            if (check_convergence and self.track_diff and self.iteration % self.convergence_check_every == 0):
                self.last_diff = self.diff(n_bins=self.diff_n_bins)

                if verbose >= 2:
                    print(
                        f"iter={self.iteration} "
                        f"diff={self.last_diff:.6g} "
                        f"stable_checks={stable_checks}/{stable_checks_required} "
                        f"samples={len(self.psi_samples)}"
                    )

                if tol > 0.0 and self.last_diff < tol:
                    stable_checks += 1
                else:
                    # the population has left a previously detected stationary regime
                    if stable:
                        if verbose >= 1:
                            print(
                                f"Population lost stability at iter={self.iteration}. "
                                "Discarding accumulated observable samples."
                            )

                        self._reset_observable_samples()

                    stable = False
                    stable_checks = 0

                if stable_checks >= stable_checks_required and not stable:
                    stable = True
                    self.diagnostics["stabilized_iteration"] = self.iteration

                if stable and not sample_observables:
                    if not sample_observables:
                        if verbose >= 1:
                            print(
                                f"Early stopping at iter={self.iteration}: "
                                f"{stable_checks_required} stable checks, "
                                f"diff={self.last_diff:.6g} < tol={tol:.6g}"
                            )
                        break

            can_sample = (sample_observables and self.iteration >= self.sampling_start_iter and self.iteration % self.sampling_interval == 0 and (
                    stable or not self.require_convergence_for_sampling or not check_convergence))

            if can_sample:
                self.update_observables(num_samples=num_samples)

                if self.last_observables_valid:  # we compute observales conditional on numerically valid samples, the full observable sample is rejected if the valid fraction falls
                    self._record_current_observables()
                else:
                    if verbose >= 2:
                        print("Observable sample rejected: "
                                f"node valid fraction={self.diagnostics['node_valid_fraction']:.6f}, "
                                f"edge valid fraction={self.diagnostics['edge_valid_fraction']:.6f}")

                if self.use_wandb:
                    self._log_wandb()

                if len(self.psi_samples) >= self.min_observable_samples:
                    break

            if self.use_wandb and self.iteration % self.log_every == 0:
                self._log_wandb()

        if sample_observables and not self.imploded and len(self.psi_samples) > 0:
            self._finalize_observable_samples()

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.time() - start_time
        self.diagnostics["runtime_sec"] = elapsed
        self.diagnostics["finished_iter"] = self.iteration
        self.diagnostics["num_observable_samples"] = len(self.psi_samples)
        self.diagnostics["stable"] = stable

        if verbose >= 1:
            msg = f"Finished at iter={self.iteration}"
            if self.last_diff is not None:
                msg += f", diff={self.last_diff:.6g}"
            msg += f", samples={len(self.psi_samples)}"
            msg += f", runtime={elapsed:.2f}s"
            print(msg)

        if self.save_final_plots or (self.use_wandb and self.wandb_log_plots):
            self._save_and_log_final_plots()

        if self.use_wandb:
            self._log_wandb(final=True)
            if finish_wandb and self.wandb_run is not None:
                self.wandb_run.finish()
                self.wandb_run = None


    @torch.inference_mode()
    def diff(self, n_bins: Optional[int] = None) -> float:
        if n_bins is None:
            n_bins = self.diff_n_bins
        old_flat = self.old_population.permute(1, 2, 0).reshape(self.K * self.K, self.M)
        new_flat = self.population.permute(1, 2, 0).reshape(self.K * self.K, self.M)
        total = torch.zeros((), dtype=self.torch_dtype, device=self.device)
        for i in range(self.K * self.K):
            old_hist = torch.histc(old_flat[i], bins=n_bins, min=0.0, max=1.0)
            new_hist = torch.histc(new_flat[i], bins=n_bins, min=0.0, max=1.0)
            total += torch.abs(new_hist - old_hist).sum()
        return float((total / self.M).item())

    #=============================================================================================================================
    # OBSERVABLES COMPUTATION AND DIAGNOSTICS
    #=============================================================================================================================

    def _node_partition_terms(self, node_messages: torch.Tensor):
        coeff = self._same_count_coefficients_torch(node_messages)
        color_weights = (coeff * self._mask_node).sum(dim=2)
        if self._has_node_field:
            color_weights.mul_(self._field_node)
        return color_weights, color_weights.sum(dim=1)


    def _observable_sample_counts(
        self,
        num_samples: Optional[int] = None,
    ) -> Tuple[int, int]:
        if num_samples is not None:
            n = int(num_samples)
            return n, n

        if self.num_samples is not None:
            n = int(self.num_samples)
            return n, n

        total_message_draws = self.observable_upsampling_factor * self.M

        node_samples = max(1, int(np.ceil(total_message_draws / self.d)))
        edge_samples = max(1, int(np.ceil(total_message_draws / 2)))

        return node_samples, edge_samples


    @torch.inference_mode()
    def update_observables(
        self,
        num_samples: Optional[int] = None,
    ) -> None:
        node_num_samples, edge_num_samples = (self._observable_sample_counts(num_samples))
        batch_size = self.observable_batch_size

        # Node contributions
        processed_node_samples = 0
        valid_node_count = 0
        total_node_count = 0
        log_node_power_sum = torch.tensor(-torch.inf, dtype=self.torch_dtype, device=self.device,)
        phi_node = torch.zeros((), dtype=self.torch_dtype, device=self.device,)  # node contribution to phi
        rho_node = torch.zeros(self.K, dtype=self.torch_dtype, device=self.device,)  # node contribution to rho

        while processed_node_samples < node_num_samples:
            current_batch = min(batch_size, node_num_samples - processed_node_samples)

            node_indices = torch.randint(low=0, high=self.M, size=(current_batch, self.d), device=self.device, generator=self.torch_generator, dtype=torch.long)
            node_messages = torch.index_select(self.population, dim=0, index=node_indices.reshape(-1)).reshape(current_batch, self.d, self.K, self.K)

            color_weights, Z_node = self._node_partition_terms(node_messages)

            valid_node = torch.isfinite(Z_node) & (Z_node > 0.0)
            logZ_valid = torch.log(Z_node[valid_node])

            if logZ_valid.numel() > 0:
                # phi node contribution for this batch:
                batch_log_weights = self.mparisi * logZ_valid
                batch_log_power_sum = torch.logsumexp(batch_log_weights, dim=0,)
                batch_weights = torch.softmax(batch_log_weights, dim=0,)
                batch_phi_node = torch.sum(batch_weights * logZ_valid)  

                # rho contribution for this batch:
                valid_color_weights = color_weights[valid_node]
                valid_Z_node = Z_node[valid_node]
                node_color_probabilities = (valid_color_weights / valid_Z_node[:, None])
                batch_rho_node = torch.sum(batch_weights[:, None] * node_color_probabilities, dim=0,)

                # merge accumulators with previous batches
                new_log_node_power_sum = torch.logaddexp(log_node_power_sum, batch_log_power_sum,)

                if torch.isneginf(log_node_power_sum):
                    phi_node = batch_phi_node
                    rho_node = batch_rho_node
                else:
                    old_weight = torch.exp(log_node_power_sum - new_log_node_power_sum)
                    batch_weight = torch.exp(batch_log_power_sum - new_log_node_power_sum)

                    phi_node = old_weight * phi_node + batch_weight * batch_phi_node
                    rho_node = old_weight * rho_node + batch_weight * batch_rho_node

                log_node_power_sum = new_log_node_power_sum
                valid_node_count += logZ_valid.numel()

            total_node_count += current_batch
            processed_node_samples += current_batch

        if valid_node_count > 0:
            logZ_node_mean = (log_node_power_sum - math.log(valid_node_count))
        else:
            logZ_node_mean = torch.tensor(-torch.inf, dtype=self.torch_dtype, device=self.device,)

        node_valid_fraction = valid_node_count / total_node_count
        self.diagnostics["node_valid_fraction"] = node_valid_fraction

        # Edge contributions
        processed_edge_samples = 0
        valid_edge_count = 0
        total_edge_count = 0
        log_edge_power_sum = torch.tensor(-torch.inf, dtype=self.torch_dtype, device=self.device,)
        phi_edge = torch.zeros((), dtype=self.torch_dtype, device=self.device,) 
        # there is no rho edge contribution

        while processed_edge_samples < edge_num_samples:
            current_batch = min(batch_size, edge_num_samples - processed_edge_samples,)

            edge_indices = torch.randint(low=0, high=self.M, size=(current_batch, 2), device=self.device, generator=self.torch_generator, dtype=torch.long,)
            msg_1 = torch.index_select(self.population, dim=0, index=edge_indices[:, 0],)
            msg_2 = torch.index_select(self.population, dim=0, index=edge_indices[:, 1],)

            Z_edge = (msg_1 * msg_2.transpose(1, 2)).sum(dim=(1, 2))

            valid_edge = (torch.isfinite(Z_edge) & (Z_edge > 0.0))
            logZ_edge_valid = torch.log(Z_edge[valid_edge])

            if logZ_edge_valid.numel() > 0:
                batch_log_weights = self.mparisi * logZ_edge_valid
                batch_log_power_sum = torch.logsumexp(batch_log_weights, dim=0,)
                batch_weights = torch.softmax(batch_log_weights,dim=0,)

                # phi edge contribution for this batch
                batch_phi_edge = torch.sum(batch_weights * logZ_edge_valid)

                # merge accumulators with previous batches
                new_log_edge_power_sum = torch.logaddexp(log_edge_power_sum, batch_log_power_sum,)

                if torch.isneginf(log_edge_power_sum):
                    phi_edge = batch_phi_edge
                else:
                    old_weight = torch.exp(log_edge_power_sum - new_log_edge_power_sum)
                    batch_weight = torch.exp(batch_log_power_sum - new_log_edge_power_sum)

                    phi_edge = old_weight * phi_edge + batch_weight * batch_phi_edge

                log_edge_power_sum = new_log_edge_power_sum
                valid_edge_count += logZ_edge_valid.numel()

            total_edge_count += current_batch
            processed_edge_samples += current_batch

        if valid_edge_count > 0:
            logZ_edge_mean = log_edge_power_sum - math.log(valid_edge_count)
        else:
            logZ_edge_mean = torch.tensor(-torch.inf, dtype=self.torch_dtype, device=self.device,)

        edge_valid_fraction = valid_edge_count / total_edge_count
        self.diagnostics["edge_valid_fraction"] = edge_valid_fraction

        # check if there are enough non degenerate samples
        self.last_observables_valid = (node_valid_fraction >= self.min_valid_fraction and edge_valid_fraction >= self.min_valid_fraction)
        if not self.last_observables_valid: 
            print("Last observables were not valid")
            print(f"Valid fraction of nodes: {node_valid_fraction}")
            print(f"Valid fraction of edges: {edge_valid_fraction}")
            self.psi = None
            self.phi = None
            self.complexity = None
            self.rho = None
            self.s = None
            return

        # compute Monte Carlo means from accumulators
        psi = logZ_node_mean - 0.5 * self.d * logZ_edge_mean
        phi = phi_node - 0.5 * self.d * phi_edge
        complexity = psi - self.mparisi * phi
        rho = rho_node

        self.psi = float(psi.item())
        self.phi = float(phi.item())
        self.complexity = float(complexity.item())
        self.rho = rho.detach().cpu().numpy()
        self.s = self.phi

        self.diagnostics["last_observable_node_samples"] = node_num_samples
        self.diagnostics["last_observable_edge_samples"] = edge_num_samples
        self.diagnostics["observable_batch_size"] = batch_size


    @torch.inference_mode()
    def population_health(self) -> Dict[str, Any]:
        pop = self.population

        finite_mask = torch.isfinite(pop)
        message_mass = pop.sum(dim=(1, 2))

        finite_messages = finite_mask.all(dim=(1, 2))
        zero_messages = message_mass <= self.torch_eps

        mean_message = pop.mean(dim=0)

        stats = {
            "iteration": int(self.iteration),
            "population_min": float(torch.nan_to_num(pop).min().item()),
            "population_max": float(torch.nan_to_num(pop).max().item()),
            "mean_message": mean_message.detach().cpu().numpy(),
            "mean_message_sum": float(mean_message.sum().item()),
            "mass_min": float(torch.nan_to_num(message_mass).min().item()),
            "mass_max": float(torch.nan_to_num(message_mass).max().item()),
            "mass_mean": float(torch.nan_to_num(message_mass).mean().item()),
            "mass_std": float(torch.nan_to_num(message_mass).std().item()),
            "max_normalization_error": float(torch.nan_to_num(torch.abs(message_mass - 1.0)).max().item()),
            "nonfinite_message_fraction": float((~finite_messages).to(self.torch_dtype).mean().item()),
            "zero_message_fraction": float(zero_messages.to(self.torch_dtype).mean().item()),
            "zero_component_fraction": float((pop <= self.torch_eps).to(self.torch_dtype).mean().item()),
        }

        return stats
    
        
    def reset_population(
            self,
            reset_iteration: bool = True,
            reset_observables: bool = True,
        ) -> None:
            initial_population = self._initialize_population()
            self.population = torch.as_tensor(initial_population, dtype=self.torch_dtype, device=self.device)
            self.old_population = self.population.clone()

            self._diag_idx = torch.arange(self.K, device=self.device)
            self._r_parent = torch.arange(self.num_parents + 1, device=self.device)
            self._r_node = torch.arange(self.d + 1, device=self.device)
            self._mask_diag = self._constraint_mask_tensor(self._r_parent + 1)
            self._mask_offdiag = self._constraint_mask_tensor(self._r_parent)
            self._mask_node = self._constraint_mask_tensor(self._r_node)
            self._field_message = torch.exp(-(
                torch.as_tensor(self.mu, dtype=self.torch_dtype, device=self.device)[:, None]
                + torch.as_tensor(self.mu, dtype=self.torch_dtype, device=self.device)[None, :]
            ) / self.d)
            self._field_node = torch.exp(-torch.as_tensor(self.mu, dtype=self.torch_dtype, device=self.device))
            self._has_message_field = bool(np.any(self.mu != 0.0))
            self._has_node_field = self._has_message_field
            self.no_update = False
            self.last_diff = None
            self.imploded = False
            self.implosion_iteration = None
            self.population_diagnostic_history = []
            self.successful_updates = 0
            self.consecutive_no_valid_updates = 0

            for key in ("imploded",
                "implosion_iteration",
                "implosion_reason",
                "implosion_discarded_fraction",
                "total_discarded_candidates",
                "total_generated_candidates",
                "last_candidate_count",
                "last_valid_candidate_count",
                "last_discarded_candidate_count",
                "last_discarded_candidate_fraction",
            ):
                self.diagnostics.pop(key, None)

            if reset_iteration:
                self.iteration = 0

            if reset_observables:
                self.psi = None
                self.phi = None
                self.complexity = None
                self.rho = None
                self.s = None

                self._reset_observable_samples()


    def _reset_observable_samples(self) -> None:
        self.psi_samples = []
        self.phi_samples = []
        self.complexity_samples = []
        self.rho_samples = []
        self.s_samples = []
        self.observable_iterations = []

        self.psi_mean = None
        self.phi_mean = None
        self.complexity_mean = None
        self.rho_mean = None
        self.s_mean = None

        self.psi_std = None
        self.phi_std = None
        self.complexity_std = None
        self.rho_std = None
        self.s_std = None

        self.psi = None
        self.phi = None
        self.complexity = None
        self.rho = None
        self.s = None

        self.last_observables_valid = False

    # -----------------------------------------------------------
    # LOGGING
    # -----------------------------------------------------------

    def _record_current_observables(self) -> None:
        if self.psi is not None:
            self.psi_samples.append(float(self.psi))
            self.observable_iterations.append(int(self.iteration))
        if self.phi is not None:
            self.phi_samples.append(float(self.phi))
        if self.complexity is not None:
            self.complexity_samples.append(float(self.complexity))
        if self.rho is not None:
            self.rho_samples.append(np.asarray(self.rho, dtype=self.dtype).copy())
        if self.s is not None:
            self.s_samples.append(float(self.s))


    def _finalize_observable_samples(self) -> None:
        if len(self.psi_samples) > 0:
            self.psi_mean = float(np.mean(self.psi_samples))
            self.psi_std = float(np.std(self.psi_samples))

        if len(self.phi_samples) > 0:
            self.phi_mean = float(np.mean(self.phi_samples))
            self.phi_std = float(np.std(self.phi_samples))

        if len(self.complexity_samples) > 0:
            self.complexity_mean = float(np.mean(self.complexity_samples))
            self.complexity_std = float(np.std(self.complexity_samples))

        if len(self.rho_samples) > 0:
            rho_arr = np.asarray(self.rho_samples, dtype=self.dtype)
            self.rho_mean = np.mean(rho_arr, axis=0)
            self.rho_std = np.std(rho_arr, axis=0)

        if len(self.s_samples) > 0:
            self.s_mean = float(np.mean(self.s_samples))
            self.s_std = float(np.std(self.s_samples))


    def _init_wandb(self) -> None:
        config = {
            "K": self.K,
            "d": self.d,
            "H": self.H,
            "problem_type": self.problem_type,
            "target_density": self.target_density.tolist(),
            "mparisi": self.mparisi,
            "mu": self.mu.tolist(),
            "M": self.M,
            "damping": self.damping,
            "num_updated": self.num_updated,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "convergence_check_every": self.convergence_check_every,
            "num_samples": self.num_samples,
            "observable_upsampling_factor": self.observable_upsampling_factor,
            "observable_batch_size": self.observable_batch_size,
            "min_observable_samples": self.min_observable_samples,
            "sampling_start_iter": self.sampling_start_iter,
            "sampling_interval": self.sampling_interval,
            "require_convergence_for_sampling": self.require_convergence_for_sampling,
            "init_type": self.init_type,
            "init_noise": self.init_noise,
            "impose_color_symmetry": self.impose_color_symmetry,
            "seed": self.seed,
            "device": str(self.device),
            "torch_dtype": str(self.torch_dtype),
            "wandb_log_plots": self.wandb_log_plots,
            "save_final_plots": self.save_final_plots,
            "final_plot_bins": self.final_plot_bins,
            "diagnostic_every": self.diagnostic_every,
            "diagnostic_hist_bins": self.diagnostic_hist_bins,
            "diagnostic_sample_size": self.diagnostic_sample_size,
            "save_diagnostic_plots": self.save_diagnostic_plots,
            "min_valid_fraction": self.min_valid_fraction,
            "max_discarded_message_fraction": self.max_discarded_message_fraction,
            "implosion_check_start_iter": self.implosion_check_start_iter,
            **self.wandb_config_extra,
        }

        self.wandb_run = wandb.init(project=self.wandb_project, group=self.wandb_group, name=self.wandb_name or self.run_name, config=config, reinit="finish_previous",)

        if self.wandb_run is None:
            raise RuntimeError(
                "wandb.init() did not create a run. Check WANDB_MODE, authentication, "
                "and network access from the compute node."
            )
        

    def _log_wandb(self, final: bool = False) -> None:
        if not self.use_wandb:
            return

        data: Dict[str, Any] = {"iteration": self.iteration, "final": final,}

        if self.last_diff is not None:
            data["diff"] = self.last_diff

        if self.psi is not None:
            data["psi"] = self.psi
        if self.phi is not None:
            data["phi"] = self.phi
        if self.complexity is not None:
            data["complexity"] = self.complexity
        if self.s is not None:
            data["s"] = self.s

        if self.rho is not None:
            for a, value in enumerate(self.rho):
                data[f"rho/{a}"] = float(value)

        if self.psi_mean is not None:
            data["psi_mean"] = self.psi_mean
            data["psi_std"] = self.psi_std
        if self.phi_mean is not None:
            data["phi_mean"] = self.phi_mean
            data["phi_std"] = self.phi_std
        if self.complexity_mean is not None:
            data["complexity_mean"] = self.complexity_mean
            data["complexity_std"] = self.complexity_std
        if self.s_mean is not None:
            data["s_mean"] = self.s_mean
            data["s_std"] = self.s_std

        if self.rho_mean is not None:
            for a, value in enumerate(self.rho_mean):
                data[f"rho_mean/{a}"] = float(value)
            for a, value in enumerate(self.rho_std):
                data[f"rho_std/{a}"] = float(value)

        if "last_observable_node_samples" in self.diagnostics:
            data["observables/node_sample_count"] = (self.diagnostics["last_observable_node_samples"])

        if "last_observable_edge_samples" in self.diagnostics:
            data["observables/edge_sample_count"] = (self.diagnostics["last_observable_edge_samples"])

        if "observable_batch_size" in self.diagnostics:
            data["observables/batch_size"] = (self.diagnostics["observable_batch_size"])

        data["observables/valid"] = int(self.last_observables_valid)
        data["observables/node_valid_fraction"] = self.diagnostics.get("node_valid_fraction")
        data["observables/edge_valid_fraction"] = self.diagnostics.get("edge_valid_fraction")

        data["population/imploded"] = int(self.imploded)
        data["population/discarded_candidate_fraction"] = self.diagnostics.get("last_discarded_candidate_fraction")
        data["population/total_discarded_candidates"] = self.diagnostics.get("total_discarded_candidates")

        data["population/valid_candidate_count"] = self.diagnostics.get("last_valid_candidate_count")
        data["population/discarded_candidate_count"] = self.diagnostics.get("last_discarded_candidate_count")
        data["population/consecutive_no_valid_updates"] = self.consecutive_no_valid_updates
        data["population/successful_updates"] = self.successful_updates

        if self.wandb_run is None:
            raise RuntimeError("wandb logging was requested, but no active wandb run exists.")
        self.wandb_run.log(data, step=self.iteration)


    def _save_and_log_final_plots(self) -> None:
        plot_dir = os.path.join(self.save_dir, self.run_name, "plots")
        if self.save_final_plots and self.save_locally:
            os.makedirs(plot_dir, exist_ok=True)

        figures: Dict[str, Any] = {}

        figures["population_histograms"] = self.plot_population_histograms(n_bins=self.final_plot_bins, save_path=(os.path.join(plot_dir, "population_histograms.png")
                if self.save_final_plots and self.save_locally else None), show=False,)
        figures["population_atoms"] = self.plot_population_atoms(decimals=12, max_atoms=200, sample_size=100_000, save_path=(os.path.join(plot_dir, "population_atoms.png",)
                if self.save_final_plots and self.save_locally else None), show=False,)
        figures["mean_message"] = self.plot_mean_message(save_path=(os.path.join(plot_dir, "mean_message.png")
                if self.save_final_plots and self.save_locally else None), show=False,)

        if len(self.population_diagnostic_history) > 0:
            figures["mean_message_evolution"] = self.plot_mean_message_evolution(save_path=(os.path.join(plot_dir, "mean_message_evolution.png")
                    if self.save_final_plots and self.save_locally else None), show=False,)
            
            figures["diagnostic_time_series"] = self.plot_diagnostic_time_series(save_path=(os.path.join(plot_dir, "diagnostic_time_series.png",)
                    if self.save_final_plots and self.save_locally else None), show=False,)

            figures["candidate_mass_evolution"] = self.plot_candidate_mass_evolution(save_path=(os.path.join(plot_dir, "candidate_mass_evolution.png",)
                    if self.save_final_plots and self.save_locally else None), show=False,)

        if len(self.psi_samples) > 0:
            figures["observable_samples"] = self.plot_observable_samples(save_path=(os.path.join(plot_dir, "observable_samples.png")
                    if self.save_final_plots and self.save_locally else None), show=False,)

        if self.use_wandb and self.wandb_log_plots:
            if self.wandb_run is None:
                raise RuntimeError("Cannot log plots because there is no active wandb run.")
            if self.use_wandb and self.wandb_log_plots:
                if self.wandb_run is None:
                    raise RuntimeError("Cannot log plots because there is no active W&B run.")

                for name, fig in figures.items():
                    self.wandb_run.log({f"plots/{name}": wandb.Image(fig),}, step=self.iteration,)

        for fig in figures.values():
            plt.close(fig)


    @torch.inference_mode()
    def _record_and_save_diagnostics(
        self,
        candidate_mass: Optional[torch.Tensor] = None,
    ) -> None:
        health = self.record_population_diagnostic(
            candidate_Z=candidate_mass
        )
        snapshot = self.diagnostic_histogram_snapshot()
        path = self.save_diagnostic_histogram(snapshot)

        print(
            f"[diagnostic iter={self.iteration}] "
            f"mass_mean={health['mass_mean']:.8g}, "
            f"mass_min={health['mass_min']:.8g}, "
            f"max_norm_error={health['max_normalization_error']:.3e}, "
            f"zero_messages={health['zero_message_fraction']:.3e}, "
            f"zero_components={health['zero_component_fraction']:.3e}, "
            f"valid_candidates="
            f"{health.get('valid_candidate_fraction', float('nan')):.3e}, "
            f"discarded_candidates="
            f"{self.diagnostics.get('last_discarded_candidate_fraction', float('nan')):.3e}, "
            f"saved={path}",
            flush=True,
        )


    @torch.inference_mode()
    def record_population_diagnostic(
        self,
        candidate_Z: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        health = self.population_health()

        health.update({
            "discarded_candidate_fraction": float(self.diagnostics.get("last_discarded_candidate_fraction",np.nan,)),
            "valid_candidate_count": int(self.diagnostics.get("last_valid_candidate_count",0,)),
            "discarded_candidate_count": int(self.diagnostics.get("last_discarded_candidate_count",0,)),
            "consecutive_no_valid_updates": int(self.consecutive_no_valid_updates),
            "successful_updates": int(self.successful_updates),
        })

        if candidate_Z is not None:
            finite_Z = torch.isfinite(candidate_Z)
            positive_Z_mask = finite_Z & (candidate_Z > self.torch_eps)
            zero_Z_mask = finite_Z & (candidate_Z <= self.torch_eps)
            nonfinite_Z_mask = ~finite_Z

            num_total = int(candidate_Z.numel())
            num_positive = int(positive_Z_mask.sum().item())
            num_zero = int(zero_Z_mask.sum().item())
            num_nonfinite = int(nonfinite_Z_mask.sum().item())

            positive_fraction = num_positive / num_total if num_total > 0 else 0.0
            zero_fraction = num_zero / num_total if num_total > 0 else 0.0
            nonfinite_fraction = num_nonfinite / num_total if num_total > 0 else 0.0

            finite_values = candidate_Z[finite_Z]

            if finite_values.numel() > 0:
                candidate_Z_min = float(finite_values.min().item())
                candidate_Z_max = float(finite_values.max().item())
                candidate_Z_mean = float(finite_values.mean().item())
            else:
                candidate_Z_min = np.nan
                candidate_Z_max = np.nan
                candidate_Z_mean = np.nan

            log10_min = math.log10(self.torch_eps)
            log10_max = 2.0
            log10_edges = np.linspace(log10_min, log10_max, self.diagnostic_hist_bins + 1,)

            if num_positive > 0:
                positive_Z = candidate_Z[positive_Z_mask]

                if positive_Z.numel() > self.diagnostic_sample_size:
                    sampled_indices = torch.randint(low=0, high=positive_Z.numel(), size=(self.diagnostic_sample_size,), device=self.device, generator=self.torch_generator,)
                    positive_Z = torch.index_select(positive_Z, dim=0, index=sampled_indices,)

                log10_Z = torch.log10(positive_Z).detach().cpu().numpy()

                log10_counts, _ = np.histogram(log10_Z, bins=log10_edges,)

                q00, q25, q50, q75, q100 = np.quantile(log10_Z, [0.0, 0.25, 0.5, 0.75, 1.0],)
            else:
                log10_counts = np.zeros(self.diagnostic_hist_bins, dtype=np.int64,)
                q00 = q25 = q50 = q75 = q100 = np.nan

            health.update({
                "candidate_Z_min": candidate_Z_min,
                "candidate_Z_max": candidate_Z_max,
                "candidate_Z_mean": candidate_Z_mean,
                "valid_candidate_fraction": positive_fraction,
                "zero_candidate_fraction": zero_fraction,
                "nonfinite_candidate_fraction": nonfinite_fraction,
                "candidate_log10_hist_counts": log10_counts,
                "candidate_log10_hist_edges": log10_edges,
                "candidate_log10_q00": float(q00),
                "candidate_log10_q25": float(q25),
                "candidate_log10_q50": float(q50),
                "candidate_log10_q75": float(q75),
                "candidate_log10_q100": float(q100),
            })

        self.population_diagnostic_history.append(health)

        if self.use_wandb and self.wandb_run is not None:
            data = {
                "diagnostics/population_min": health["population_min"],
                "diagnostics/population_max": health["population_max"],
                "diagnostics/mean_message_sum": health["mean_message_sum"],
                "diagnostics/mass_min": health["mass_min"],
                "diagnostics/mass_max": health["mass_max"],
                "diagnostics/mass_mean": health["mass_mean"],
                "diagnostics/max_normalization_error": health["max_normalization_error"],
                "diagnostics/nonfinite_message_fraction": health["nonfinite_message_fraction"],
                "diagnostics/zero_message_fraction": health["zero_message_fraction"],
                "diagnostics/zero_component_fraction": health["zero_component_fraction"],
                "diagnostics/discarded_candidate_fraction": health["discarded_candidate_fraction"],
                "diagnostics/valid_candidate_count": health["valid_candidate_count"],
                "diagnostics/discarded_candidate_count": health["discarded_candidate_count"],
                "diagnostics/consecutive_no_valid_updates": health["consecutive_no_valid_updates"],
                "diagnostics/successful_updates": health["successful_updates"],
            }

            for key in (
                "valid_candidate_fraction",
                "zero_candidate_fraction",
                "nonfinite_candidate_fraction",
                "candidate_Z_min",
                "candidate_Z_max",
                "candidate_Z_mean",
                "candidate_log10_q00",
                "candidate_log10_q25",
                "candidate_log10_q50",
                "candidate_log10_q75",
                "candidate_log10_q100",
            ):
                if key in health:
                    data[f"diagnostics/{key}"] = health[key]

            self.wandb_run.log(data, step=self.iteration,)

        return health
        

    def save_diagnostic_histogram(
        self,
        snapshot: Dict[str, Any],
    ) -> Optional[str]:
        if not self.save_diagnostic_plots:
            return None

        diagnostic_dir = os.path.join(self.save_dir, self.run_name, "diagnostics",)
        os.makedirs(diagnostic_dir, exist_ok=True)

        iteration = snapshot["iteration"]

        save_path = os.path.join(diagnostic_dir, f"population_hist_iter_{iteration:07d}.png",)

        fig = self.plot_population_histogram_snapshot(
            snapshot,
            title=(
                f"Population diagnostic at iteration {iteration}\n"
                f"{self._default_title()}"
            ),
            save_path=save_path,
            show=False,
        )

        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.log({"diagnostics/population_histogram": wandb.Image(fig),}, step=iteration,)

        plt.close(fig)
        return save_path
        
    #=============================================================================================================================
    # Plotting
    #=============================================================================================================================

    def _population_numpy(self, old: bool = False) -> np.ndarray:
        pop = self.old_population if old else self.population
        return pop.detach().cpu().numpy()


    def _default_title(self) -> str:
        return (
            f"{self.problem_type}, "
            f"K={self.K}, d={self.d}, H={self.H}, "
            f"m={self.mparisi:.4g}, M={self.M}"
        )


    def _prepare_save_path(
        self,
        save_path: Optional[str],
    ) -> Optional[str]:
        if save_path is None:
            return None

        directory = os.path.dirname(save_path)
        if directory != "":
            os.makedirs(directory, exist_ok=True)

        return save_path


    def _finalize_plot(
        self,
        fig,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        save_path = self._prepare_save_path(save_path)

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=200)

        if show:
            plt.show()

        return fig


    def plot_population_histograms(
        self,
        n_bins: int = 100,
        old: bool = False,
        density: bool = True,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        pop = self._population_numpy(old=old)

        if title is None:
            which = "old population" if old else "population"
            title = f"{which} histograms\n{self._default_title()}"

        fig, axes = plt.subplots(self.K, self.K, figsize=(2.8 * self.K, 2.4 * self.K), squeeze=False,)

        for x in range(self.K):
            for y in range(self.K):
                ax = axes[x, y]

                bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

                ax.hist(pop[:, x, y], bins=bin_edges, density=density, rwidth=1.0,)

                mean_value = np.mean(pop[:, x, y])
                ax.axvline(mean_value, linestyle="--")
                ax.text(0.98, 0.92, f"mean={mean_value:.4g}", transform=ax.transAxes, ha="right", va="top",)

                ax.set_xlim(0.0, 1.0)
                ax.set_title(rf"$\chi_{{{x},{y}}}$")
                ax.set_xlabel("value")
                ax.set_ylabel("density" if density else "count")

        fig.suptitle(title)
        fig.tight_layout()

        return self._finalize_plot(fig, save_path=save_path, show=show)


    def plot_mean_message(
        self,
        old: bool = False,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        pop = self._population_numpy(old=old)
        mean_chi = np.mean(pop, axis=0)

        if title is None:
            which = "old mean message" if old else "mean message"
            title = f"{which}\n{self._default_title()}"

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(mean_chi, aspect="equal")
        fig.colorbar(im, ax=ax)

        ax.set_title(title)
        ax.set_xlabel(r"neighbour colour $y$")
        ax.set_ylabel(r"central colour $x$")
        ax.set_xticks(np.arange(self.K))
        ax.set_yticks(np.arange(self.K))

        for x in range(self.K):
            for y in range(self.K):
                ax.text(y, x, f"{mean_chi[x, y]:.3g}", ha="center", va="center",)

        fig.tight_layout()
        return self._finalize_plot(fig, save_path=save_path, show=show)
    

    def plot_observable_samples(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        plot_mean: bool = True,
        plot_sem_band: bool = True,
    ):
        if len(self.psi_samples) == 0:
            raise ValueError(
                "No observable samples available. "
                "This usually means the run did not reach the stability threshold, "
                "or require_convergence_for_sampling=True prevented sampling."
            )

        if hasattr(self, "observable_iterations") and len(self.observable_iterations) == len(self.psi_samples):
            t = np.asarray(self.observable_iterations)
            xlabel = "iteration"
        else:
            t = np.arange(len(self.psi_samples))
            xlabel = "observable sample index"

        fig, axes = plt.subplots(5, 1, figsize=(8, 12), sharex=True)

        def _plot_series(ax, values, ylabel, draw_zero_line=False):
            values = np.asarray(values, dtype=float)

            ax.plot(t, values, marker="o")
            ax.set_ylabel(ylabel)

            if draw_zero_line:
                ax.axhline(0.0, linestyle="--")

            if plot_mean:
                mean_val = np.mean(values)
                std_val = np.std(values)
                sem_val = std_val / np.sqrt(len(values))

                ax.axhline(mean_val, linestyle="--")

                if plot_sem_band:
                    ax.fill_between(t, mean_val - sem_val, mean_val + sem_val, alpha=0.2,)

                ax.text(0.99, 0.92, f"mean={mean_val:.4g}\nsem={sem_val:.3g}", transform=ax.transAxes, ha="right", va="top",)

        _plot_series(axes[0], self.psi_samples, r"$\Psi$")
        _plot_series(axes[1], self.phi_samples, r"$\phi$")
        _plot_series(axes[2], self.complexity_samples, r"$\Sigma$", draw_zero_line=True)
        _plot_series(axes[3], self.s_samples, r"$s$")

        rho_arr = np.asarray(self.rho_samples, dtype=float)

        for a in range(self.K):
            values = rho_arr[:, a]
            axes[4].plot(t, values, marker="o", label=rf"$\rho_{a}$")

            if plot_mean:
                mean_val = np.mean(values)
                std_val = np.std(values)
                sem_val = std_val / np.sqrt(len(values))

                axes[4].axhline(mean_val, linestyle="--")

                if plot_sem_band:
                    axes[4].fill_between(t, mean_val - sem_val, mean_val + sem_val, alpha=0.15,)

        axes[4].axhline(1.0 / self.K, linestyle="--")
        axes[4].set_ylabel(r"$\rho$")
        axes[4].set_xlabel(xlabel)
        axes[4].legend()

        if title is None:
            title = f"Observable samples\n{self._default_title()}"

        fig.suptitle(title)
        fig.tight_layout()

        return self._finalize_plot(fig, save_path=save_path, show=show)
    

    def compute_complexity_curves(
        self,
        m_list: Optional[np.ndarray] = None,
        reset_population_each_m: bool = True,
        check_convergence: bool = True,
        verbose: int = 1,
        **run_kwargs,
    ) -> None:
        if m_list is not None:
            self.m_list = np.asarray(m_list, dtype=self.dtype)

        if self.m_list is None:
            raise ValueError("m_list is None. Provide m_list or set self.m_list first.")

        num_m = len(self.m_list)

        self.psi_list = np.zeros(num_m, dtype=self.dtype)
        self.psi_list_std = np.zeros(num_m, dtype=self.dtype)

        self.phi_list = np.zeros(num_m, dtype=self.dtype)
        self.phi_list_std = np.zeros(num_m, dtype=self.dtype)

        self.complexity_list = np.zeros(num_m, dtype=self.dtype)
        self.complexity_list_std = np.zeros(num_m, dtype=self.dtype)

        self.rho_list = np.zeros((num_m, self.K), dtype=self.dtype)
        self.rho_list_std = np.zeros((num_m, self.K), dtype=self.dtype)

        self.s_list = np.zeros(num_m, dtype=self.dtype)
        self.s_list_std = np.zeros(num_m, dtype=self.dtype)

        for i, m in enumerate(self.m_list):
            if verbose >= 1:
                print(f"========== m = {m:.6g} ==========")

            self.mparisi = float(m)

            if reset_population_each_m:
                self.reset_population(reset_iteration=True, reset_observables=True)
            else:
                self._reset_observable_samples()

            self.run(check_convergence=check_convergence, verbose=verbose, **run_kwargs,)

            self.psi_list[i] = self.psi_mean
            self.psi_list_std[i] = self.psi_std

            self.phi_list[i] = self.phi_mean
            self.phi_list_std[i] = self.phi_std

            self.complexity_list[i] = self.complexity_mean
            self.complexity_list_std[i] = self.complexity_std

            self.rho_list[i] = self.rho_mean
            self.rho_list_std[i] = self.rho_std

            self.s_list[i] = self.s_mean
            self.s_list_std[i] = self.s_std


    def plot_complexity_vs_phi(
        self,
        errorbars: bool = True,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        if self.phi_list is None or self.complexity_list is None:
            raise ValueError("No complexity curve available. Run compute_complexity_curves first.")

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.axhline(0.0, linestyle="--")

        if errorbars and self.phi_list_std is not None and self.complexity_list_std is not None:
            ax.errorbar(self.phi_list, self.complexity_list, xerr=self.phi_list_std, yerr=self.complexity_list_std, fmt="o", capsize=3,)
        else:
            ax.plot(self.phi_list, self.complexity_list, marker="o")

        ax.set_xlabel(r"$\phi_{\mathrm{int}}$")
        ax.set_ylabel(r"$\Sigma$")
        ax.set_title(title or f"Complexity curve\n{self._default_title()}")
        fig.tight_layout()

        return self._finalize_plot(fig, save_path=save_path, show=show)
    

    def plot_complexity_vs_m(
        self,
        errorbars: bool = True,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        if self.m_list is None or self.complexity_list is None:
            raise ValueError("No complexity curve available. Run compute_complexity_curves first.")

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.axhline(0.0, linestyle="--")

        if errorbars and self.complexity_list_std is not None:
            ax.errorbar(self.m_list, self.complexity_list, yerr=self.complexity_list_std, fmt="o", capsize=3,)
        else:
            ax.plot(self.m_list, self.complexity_list, marker="o")

        ax.set_xlabel(r"$m$")
        ax.set_ylabel(r"$\Sigma$")
        ax.set_title(title or f"Complexity versus m\n{self._default_title()}")
        fig.tight_layout()

        return self._finalize_plot(fig, save_path=save_path, show=show)
    

    def plot_complexity_vs_rho_component(
        self,
        component: int = 0,
        errorbars: bool = True,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        if self.rho_list is None or self.complexity_list is None:
            raise ValueError("No complexity curve available. Run compute_complexity_curves first.")

        if component < 0 or component >= self.K:
            raise ValueError(f"component must be in [0, {self.K - 1}].")

        rho_component = self.rho_list[:, component]
        rho_component_std = self.rho_list_std[:, component]

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.axhline(0.0, linestyle="--")
        ax.axvline(1.0 / self.K, linestyle="--")

        if errorbars:
            ax.errorbar(rho_component, self.complexity_list, xerr=rho_component_std, yerr=self.complexity_list_std, fmt="o", capsize=3,)
        else:
            ax.plot(rho_component, self.complexity_list, marker="o")

        ax.set_xlabel(rf"$\rho_{component}$")
        ax.set_ylabel(r"$\Sigma$")
        ax.set_title(title or f"Complexity versus density component\n{self._default_title()}")
        fig.tight_layout()

        return self._finalize_plot(fig, save_path=save_path, show=show)


    def plot_complexity_vs_balance_error(
        self,
        errorbars: bool = False,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        if self.rho_list is None or self.complexity_list is None:
            raise ValueError("No complexity curve available. Run compute_complexity_curves first.")

        target = np.full(self.K, 1.0 / self.K, dtype=self.dtype)
        balance_error = np.linalg.norm(self.rho_list - target[None, :], axis=1)

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.axhline(0.0, linestyle="--")

        if errorbars:
            ax.errorbar(balance_error, self.complexity_list, yerr=self.complexity_list_std, fmt="o", capsize=3,)
        else:
            ax.plot(balance_error, self.complexity_list, marker="o")

        ax.set_xlabel(r"$\|\rho - (1/K,\ldots,1/K)\|_2$")
        ax.set_ylabel(r"$\Sigma$")
        ax.set_title(title or f"Complexity versus balance error\n{self._default_title()}")
        fig.tight_layout()

        return self._finalize_plot(fig, save_path=save_path, show=show)
    

    def population_histogram_snapshot(
        self,
        n_bins: int = 80,
        old: bool = False,
        density: bool = True,
    ) -> Dict[str, Any]:
        pop = self._population_numpy(old=old)

        counts = np.zeros((self.K, self.K, n_bins), dtype=self.dtype)
        edges = None

        for x in range(self.K):
            for y in range(self.K):
                hist, bin_edges = np.histogram(pop[:, x, y], bins=n_bins, range=(0.0, 1.0), density=density,)
                counts[x, y] = hist

                if edges is None:
                    edges = bin_edges

        return {
            "iteration": self.iteration,
            "counts": counts,
            "edges": edges,
            "density": density,
        }


    def plot_population_histogram_snapshot(
        self,
        snapshot: Dict[str, Any],
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        counts = snapshot["counts"]
        edges = snapshot["edges"]
        density = snapshot["density"]
        iteration = snapshot["iteration"]

        centers = 0.5 * (edges[:-1] + edges[1:])

        fig, axes = plt.subplots(self.K, self.K, figsize=(2.8 * self.K, 2.4 * self.K), squeeze=False,)

        ymax = np.max(counts)
        if ymax <= 0:
            ymax = 1.0

        for x in range(self.K):
            for y in range(self.K):
                ax = axes[x, y]
                ax.bar(centers, counts[x, y], width=edges[1] - edges[0], align="center",)
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.05 * ymax)
                ax.set_title(rf"$\chi_{{{x},{y}}}$")
                ax.set_xlabel("value")
                ax.set_ylabel("density" if density else "count")

        if title is None:
            title = f"Population histogram snapshot, iter={iteration}\n{self._default_title()}"

        fig.suptitle(title)
        fig.tight_layout()

        return self._finalize_plot(fig, save_path=save_path, show=show)
    

    def plot_population_atoms(
        self,
        decimals: int = 12,
        old: bool = False,
        max_atoms: int = 200,
        sample_size: int = 100_000,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        pop_tensor = self.old_population if old else self.population

        sample_size = min(int(sample_size), self.M)

        if sample_size < self.M:
            indices = torch.randint(low=0, high=self.M, size=(sample_size,), device=self.device, generator=self.torch_generator, dtype=torch.long,)

            pop = torch.index_select(pop_tensor, dim=0, index=indices,).detach().cpu().numpy()
        else:
            pop = pop_tensor.detach().cpu().numpy()

        if title is None:
            which = "old population" if old else "population"
            title = (
                f"{which} approximate atoms\n"
                f"{self._default_title()}"
            )

        fig, axes = plt.subplots(self.K, self.K, figsize=(2.8 * self.K, 2.4 * self.K), squeeze=False,)

        for x in range(self.K):
            for y in range(self.K):
                ax = axes[x, y]

                component = pop[:, x, y]
                rounded = np.round(component, decimals=decimals)

                unique, counts = np.unique(rounded, return_counts=True,)

                total_unique = len(unique)

                if total_unique > max_atoms:
                    most_frequent = np.argpartition(counts, -max_atoms,)[-max_atoms:]

                    unique = unique[most_frequent]
                    counts = counts[most_frequent]

                    order = np.argsort(unique)
                    unique = unique[order]
                    counts = counts[order]

                if len(unique) > 1:
                    sorted_unique = np.sort(unique)
                    separations = np.diff(sorted_unique)
                    positive_separations = separations[separations > 0]

                    if len(positive_separations) > 0:
                        width = min(0.015, 0.8 * np.min(positive_separations),)
                    else:
                        width = 0.015
                else:
                    width = 0.015

                ax.bar(unique, counts, width=width, align="center",)

                mean_value = float(np.mean(component))
                ax.axvline(mean_value, linestyle="--")

                ax.set_xlim(-0.02, 1.02)
                ax.set_title(rf"$\chi_{{{x},{y}}}$")
                ax.set_xlabel("rounded value")
                ax.set_ylabel("sample count")

                text = (
                    f"mean={mean_value:.4g}\n"
                    f"sample={sample_size:,}\n"
                    f"unique={total_unique:,}"
                )

                if total_unique > max_atoms:
                    text += f"\nshowing top {max_atoms}"

                ax.text(0.98, 0.95, text, transform=ax.transAxes, ha="right", va="top", fontsize=8,)

        fig.suptitle(title)
        fig.tight_layout()

        return self._finalize_plot(fig, save_path=save_path, show=show,)
    

    @torch.inference_mode()
    def diagnostic_histogram_snapshot(
        self,
        n_bins: Optional[int] = None,
        sample_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        if n_bins is None:
            n_bins = self.diagnostic_hist_bins

        if sample_size is None:
            sample_size = self.diagnostic_sample_size

        sample_size = min(int(sample_size), self.M)

        indices = torch.randint(low=0, high=self.M, size=(sample_size,), device=self.device, generator=self.torch_generator,)

        sampled_population = torch.index_select(self.population, dim=0, index=indices,).detach().cpu().numpy()

        counts = np.zeros((self.K, self.K, n_bins), dtype=np.int64,)

        edges = np.linspace(0.0, 1.0, n_bins + 1)

        for x in range(self.K):
            for y in range(self.K):
                counts[x, y], _ = np.histogram(sampled_population[:, x, y], bins=edges,)

        return {
            "iteration": int(self.iteration),
            "counts": counts,
            "edges": edges,
            "sample_size": sample_size,
            "mean_message": sampled_population.mean(axis=0),
            "message_mass": sampled_population.sum(axis=(1, 2)),
            "density": False,
        }
    

    def plot_candidate_mass_evolution(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        history = [item for item in self.population_diagnostic_history if "candidate_log10_hist_counts" in item]

        if len(history) == 0:
            raise ValueError("No candidate-mass diagnostics have been recorded.")

        iterations = np.asarray([item["iteration"] for item in history])

        counts = np.stack([np.asarray(item["candidate_log10_hist_counts"], dtype=float,) for item in history])

        edges = np.asarray(history[0]["candidate_log10_hist_edges"], dtype=float,)
        centers = 0.5 * (edges[:-1] + edges[1:])

        fig, ax = plt.subplots(figsize=(10, 6))

        if np.all(counts == 0):
            ax.text(0.5, 0.5, "No positive candidate masses\n" 
                    "were observed in any diagnostic snapshot.", transform=ax.transAxes, ha="center", va="center",)
            ax.set_xlabel("iteration")
            ax.set_ylabel(r"$\log_{10}(Z_{\mathrm{candidate}})$")
        else:
            mesh = ax.pcolormesh(iterations, centers, np.log1p(counts.T), shading="nearest",)

            colorbar = fig.colorbar(mesh, ax=ax,)
            colorbar.set_label(r"$\log(1+\mathrm{count})$")

            ax.set_xlabel("iteration")
            ax.set_ylabel(r"$\log_{10}(Z_{\mathrm{candidate}})$")

        ax.set_title(
            f"Candidate-mass evolution\n"
            f"{self._default_title()}"
        )

        fig.tight_layout()

        return self._finalize_plot(fig, save_path=save_path, show=show,)
    
    def plot_diagnostic_time_series(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        if len(self.population_diagnostic_history) == 0:
            raise ValueError("No population diagnostics have been recorded.")

        history = self.population_diagnostic_history

        iterations = np.asarray([item["iteration"] for item in history])
        discarded_fraction = np.asarray([item.get("discarded_candidate_fraction", np.nan) for item in history])
        valid_candidate_count = np.asarray([item.get("valid_candidate_count", np.nan) for item in history])
        zero_candidate_fraction = np.asarray([item.get("zero_candidate_fraction", np.nan) for item in history])
        nonfinite_candidate_fraction = np.asarray([item.get("nonfinite_candidate_fraction", np.nan) for item in history])
        consecutive_no_valid = np.asarray([item.get("consecutive_no_valid_updates", np.nan) for item in history]) 
        successful_updates = np.asarray([item.get("successful_updates", np.nan) for item in history])

        fig, axes = plt.subplots(5, 1, figsize=(9, 13), sharex=True,)

        axes[0].plot(iterations, discarded_fraction, marker="o",)
        axes[0].axhline(self.max_discarded_message_fraction, linestyle="--", label="implosion threshold",)
        axes[0].set_ylabel("discarded fraction")
        axes[0].set_ylim(-0.02, 1.02)
        axes[0].legend()

        axes[1].plot(iterations, valid_candidate_count, marker="o",)
        axes[1].set_ylabel("valid candidates")
        axes[1].set_yscale("symlog", linthresh=1.0,)

        axes[2].plot(iterations, zero_candidate_fraction, marker="o", label="zero mass",)
        axes[2].plot(iterations, nonfinite_candidate_fraction, marker="o", label="nonfinite",)
        axes[2].set_ylabel("candidate fraction")
        axes[2].set_ylim(-0.02, 1.02)
        axes[2].legend()

        axes[3].plot(iterations, consecutive_no_valid, marker="o",)
        axes[3].set_ylabel("consecutive\nno-valid updates")

        axes[4].plot(iterations, successful_updates, marker="o", )
        axes[4].set_ylabel("successful updates")
        axes[4].set_xlabel("iteration")

        fig.suptitle(
            f"Population-dynamics diagnostics\n"
            f"{self._default_title()}"
        )
        fig.tight_layout()

        return self._finalize_plot(fig, save_path=save_path, show=show,)
    
    
    def plot_mean_message_evolution(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        if len(self.population_diagnostic_history) == 0:
            raise ValueError("No population diagnostics have been recorded.")

        iterations = np.asarray([item["iteration"] for item in self.population_diagnostic_history])

        mean_messages = np.stack([item["mean_message"] for item in self.population_diagnostic_history])

        fig, ax = plt.subplots(figsize=(9, 5))

        for x in range(self.K):
            for y in range(self.K):
                ax.plot(iterations, mean_messages[:, x, y], marker="o", label=rf"$\chi_{{{x},{y}}}$",)

        ax.set_xlabel("iteration")
        ax.set_ylabel("population mean")
        ax.set_title(f"Mean-message evolution\n{self._default_title()}")
        ax.legend(ncol=min(4, self.K), fontsize=8, bbox_to_anchor=(1.02, 1.0), loc="upper left",)

        fig.tight_layout()

        return self._finalize_plot(fig, save_path=save_path, show=show,)