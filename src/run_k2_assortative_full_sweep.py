#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

from population_dynamics_optimized import PopDyn


# All non-UNSAT points shown in the K=2 assortative phase diagram.
# Filled upward triangles are labelled easy_RS; open downward triangles
# are labelled frozen_1RSB, consistently with the previously tested points.
CONFIGS = [
    # d = 3
    {"d": 3, "H": 1, "expected_phase": "easy_RS"},
    {"d": 3, "H": 2, "expected_phase": "easy_RS"},

    # d = 4
    {"d": 4, "H": 1, "expected_phase": "easy_RS"},
    {"d": 4, "H": 2, "expected_phase": "easy_RS"},

    # d = 5
    {"d": 5, "H": 1, "expected_phase": "easy_RS"},
    {"d": 5, "H": 2, "expected_phase": "easy_RS"},
    {"d": 5, "H": 3, "expected_phase": "easy_RS"},

    # d = 6
    {"d": 6, "H": 1, "expected_phase": "easy_RS"},
    {"d": 6, "H": 2, "expected_phase": "easy_RS"},
    {"d": 6, "H": 3, "expected_phase": "easy_RS"},

    # d = 7
    {"d": 7, "H": 1, "expected_phase": "easy_RS"},
    {"d": 7, "H": 2, "expected_phase": "easy_RS"},
    {"d": 7, "H": 3, "expected_phase": "easy_RS"},
    {"d": 7, "H": 4, "expected_phase": "easy_RS"},

    # d = 8
    {"d": 8, "H": 1, "expected_phase": "easy_RS"},
    {"d": 8, "H": 2, "expected_phase": "easy_RS"},
    {"d": 8, "H": 3, "expected_phase": "easy_RS"},
    {"d": 8, "H": 4, "expected_phase": "easy_RS"},
    {"d": 8, "H": 5, "expected_phase": "frozen_1RSB"},

    # d = 9
    {"d": 9, "H": 1, "expected_phase": "easy_RS"},
    {"d": 9, "H": 2, "expected_phase": "easy_RS"},
    {"d": 9, "H": 3, "expected_phase": "easy_RS"},
    {"d": 9, "H": 4, "expected_phase": "easy_RS"},
    {"d": 9, "H": 5, "expected_phase": "frozen_1RSB"},

    # d = 10
    {"d": 10, "H": 1, "expected_phase": "easy_RS"},
    {"d": 10, "H": 2, "expected_phase": "easy_RS"},
    {"d": 10, "H": 3, "expected_phase": "easy_RS"},
    {"d": 10, "H": 4, "expected_phase": "easy_RS"},
    {"d": 10, "H": 5, "expected_phase": "frozen_1RSB"},
    {"d": 10, "H": 6, "expected_phase": "frozen_1RSB"},

    # d = 11
    {"d": 11, "H": 1, "expected_phase": "easy_RS"},
    {"d": 11, "H": 2, "expected_phase": "easy_RS"},
    {"d": 11, "H": 3, "expected_phase": "easy_RS"},
    {"d": 11, "H": 4, "expected_phase": "easy_RS"},
    {"d": 11, "H": 5, "expected_phase": "frozen_1RSB"},
    {"d": 11, "H": 6, "expected_phase": "frozen_1RSB"},

    # d = 12
    {"d": 12, "H": 1, "expected_phase": "easy_RS"},
    {"d": 12, "H": 2, "expected_phase": "easy_RS"},
    {"d": 12, "H": 3, "expected_phase": "easy_RS"},
    {"d": 12, "H": 4, "expected_phase": "easy_RS"},
    {"d": 12, "H": 5, "expected_phase": "frozen_1RSB"},
    {"d": 12, "H": 6, "expected_phase": "frozen_1RSB"},
    {"d": 12, "H": 7, "expected_phase": "frozen_1RSB"},
]


def scalar_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def summarize_run(pd: PopDyn, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    rho_mean = None if pd.rho_mean is None else np.asarray(pd.rho_mean, dtype=float)
    rho_std = None if pd.rho_std is None else np.asarray(pd.rho_std, dtype=float)

    return {
        "config_index": CONFIGS.index(cfg),
        "d": cfg["d"],
        "H": cfg["H"],
        "expected_phase": cfg["expected_phase"],
        "seed": seed,
        "mparisi": float(pd.mparisi),
        "M": int(pd.M),
        "psi_mean": scalar_or_nan(pd.psi_mean),
        "psi_std": scalar_or_nan(pd.psi_std),
        "phi_mean": scalar_or_nan(pd.phi_mean),
        "phi_std": scalar_or_nan(pd.phi_std),
        "complexity_mean": scalar_or_nan(pd.complexity_mean),
        "complexity_std": scalar_or_nan(pd.complexity_std),
        "s_mean": scalar_or_nan(pd.s_mean),
        "s_std": scalar_or_nan(pd.s_std),
        "rho0_mean": float(rho_mean[0]) if rho_mean is not None else float("nan"),
        "rho1_mean": float(rho_mean[1]) if rho_mean is not None else float("nan"),
        "rho0_std": float(rho_std[0]) if rho_std is not None else float("nan"),
        "rho1_std": float(rho_std[1]) if rho_std is not None else float("nan"),
        "balance_error": (
            float(np.linalg.norm(rho_mean - 0.5))
            if rho_mean is not None else float("nan")
        ),
        "last_diff": scalar_or_nan(pd.last_diff),
        "finished_iter": int(pd.iteration),
        "num_observable_samples": len(pd.psi_samples),
        "stable": bool(pd.diagnostics.get("stable", False)),
        "runtime_sec": float(pd.diagnostics.get("runtime_sec", float("nan"))),
    }


def print_population_diagnostics(pd: PopDyn) -> None:
    pop = pd.population
    with torch.no_grad():
        mean_message = pop.mean(dim=0).detach().cpu().numpy()
        std_message = pop.std(dim=0, unbiased=False).detach().cpu().numpy()
        pop_min = float(pop.min().item())
        pop_max = float(pop.max().item())
        max_sum_error = float(
            torch.max(torch.abs(pop.sum(dim=(1, 2)) - 1.0)).item()
        )

    print("Mean message:")
    print(mean_message)
    print("Std message:")
    print(std_message)
    print(f"Population min/max: {pop_min:.8g}, {pop_max:.8g}")
    print(f"Maximum message-normalization error: {max_sum_error:.3e}")


def run_one(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    d = int(cfg["d"])
    H = int(cfg["H"])
    expected_phase = str(cfg["expected_phase"])

    M = int(os.environ.get("PD_M", "1000000"))
    max_iter = int(os.environ.get("PD_MAX_ITER", "20000"))
    obs_factor = int(os.environ.get("PD_OBS_FACTOR", "100"))
    min_obs_samples = int(os.environ.get("PD_MIN_OBS_SAMPLES", "20"))

    run_name = (
        f"K2_assortative_d{d}_H{H}_{expected_phase}"
        f"_m1_M{M}_seed{seed}"
    )

    pd = PopDyn(
        K=2,
        d=d,
        H=H,
        problem_type="assortative",
        mparisi=1.0,

        M=M,
        damping=0.8,
        max_iter=max_iter,

        tol=0.015,
        convergence_check_every=200,
        diff_n_bins=200,

        sampling_start_iter=5_000,
        sampling_interval=500,
        min_observable_samples=min_obs_samples,
        observable_upsampling_factor=obs_factor,
        require_convergence_for_sampling=True,

        init_type="hard_field",
        impose_color_symmetry=True,
        seed=seed,

        device="cuda",
        dtype=np.float64,

        use_wandb=True,
        wandb_project="assortative-popdyn",
        wandb_group="K2_full_nonUNSAT_sweep",
        wandb_name=run_name,
        log_every=200,
        wandb_config_extra={
            "paper": "Behrens_Arpino_Kivva_Zdeborova_2022",
            "target": "K2_assortative_nonUNSAT_phase_diagram",
            "expected_phase_from_figure": expected_phase,
            "config_index": CONFIGS.index(cfg),
        },

        # Let PopDyn create and upload the same final plots.
        wandb_log_plots=True,
        save_final_plots=True,
        final_plot_bins=200,
        save_locally=True,
        save_dir="results/k2_full_sweep",
        run_name=run_name,
    )

    try:
        # Keep wandb open so the scalar summary can be added afterwards.
        pd.run(
            verbose=2,
            stable_checks_required=5,
            finish_wandb=False,
        )

        print(pd)
        print_population_diagnostics(pd)

        summary = summarize_run(pd, cfg, seed)

        if pd.use_wandb and pd.wandb_run is not None:
            numeric_summary = {
                f"summary/{key}": value
                for key, value in summary.items()
                if isinstance(value, (int, float, bool, np.integer, np.floating))
            }
            pd.wandb_run.log(numeric_summary, step=pd.iteration)
            for key, value in summary.items():
                pd.wandb_run.summary[key] = value

        return summary

    finally:
        if pd.use_wandb and pd.wandb_run is not None:
            pd.wandb_run.finish()


def write_result(summary: dict[str, Any], output_root: Path) -> None:
    result_dir = output_root / "summaries"
    result_dir.mkdir(parents=True, exist_ok=True)

    d = summary["d"]
    H = summary["H"]
    seed = summary["seed"]

    json_path = result_dir / f"d{d}_H{H}_seed{seed}.json"
    csv_path = result_dir / f"d{d}_H{H}_seed{seed}.csv"

    with json_path.open("w") as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


def main() -> None:
    task_id_text = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id_text is None:
        raise RuntimeError(
            "SLURM_ARRAY_TASK_ID is not set. Submit this script as a Slurm array "
            f"with indices 0-{len(CONFIGS) - 1}."
        )

    task_id = int(task_id_text)
    if task_id < 0 or task_id >= len(CONFIGS):
        raise IndexError(
            f"Task index {task_id} is outside 0-{len(CONFIGS) - 1}."
        )

    seed = int(os.environ.get("PD_SEED", "42"))
    cfg = CONFIGS[task_id]

    print(f"Running configuration {task_id + 1}/{len(CONFIGS)}: {cfg}")
    summary = run_one(cfg, seed=seed)
    write_result(summary, Path("results/k2_full_sweep"))


if __name__ == "__main__":
    main()
