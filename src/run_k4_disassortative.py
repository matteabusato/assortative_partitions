#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# Make imports independent of the directory from which the script is launched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from population_dynamics_optimized import PopDyn


# Trial configurations for K=4 disassortative population dynamics.
# The Slurm array index selects one entry from this list.
CONFIGS: List[Dict[str, Any]] = [
    {"d": 12, "H": 2, "label": "d12_H2"},
    {"d": 12, "H": 3, "label": "d12_H3"},
    {"d": 12, "H": 4, "label": "d12_H4"},
    {"d": 12, "H": 5, "label": "d12_H5"},
    {"d": 12, "H": 6, "label": "d12_H6"},
    {"d": 12, "H": 7, "label": "d12_H7"},
]


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def select_config() -> tuple[int, Dict[str, Any]]:
    raw_index = os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("PD_CONFIG_INDEX", "0"))
    index = int(raw_index)

    if index < 0 or index >= len(CONFIGS):
        raise IndexError(
            f"Configuration index {index} is outside [0, {len(CONFIGS) - 1}]."
        )

    return index, CONFIGS[index]


def write_summary(summary: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = (
        f"d{summary['d']}_H{summary['H']}"
        f"_seed{summary['seed']}"
    )

    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    flat_summary = {
        key: value
        for key, value in summary.items()
        if not isinstance(value, (list, dict))
    }

    for key, value in summary.items():
        if isinstance(value, list):
            for i, item in enumerate(value):
                flat_summary[f"{key}_{i}"] = item

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_summary.keys()))
        writer.writeheader()
        writer.writerow(flat_summary)

    print(f"Saved {json_path}", flush=True)
    print(f"Saved {csv_path}", flush=True)


def main() -> None:
    config_index, cfg = select_config()

    K = 4
    d = int(cfg["d"])
    H = int(cfg["H"])

    seed = env_int("PD_SEED", 44)
    M = env_int("PD_M", 1_000_000)
    max_iter = env_int("PD_MAX_ITER", 3_000)
    raw_num_samples = os.environ.get("PD_NUM_SAMPLES")

    num_samples = (
        None
        if raw_num_samples in (None, "", "none", "None")
        else int(raw_num_samples)
    )

    observable_batch_size = env_int(
        "PD_OBSERVABLE_BATCH_SIZE",
        20_000,
    )


    damping = env_float("PD_DAMPING", 0.8)
    tol = env_float("PD_TOL", 0.015)
    mparisi = env_float("PD_MPARISI", 1.0)

    convergence_check_every = env_int("PD_CONVERGENCE_CHECK_EVERY", 200)
    stable_checks_required = env_int("PD_STABLE_CHECKS", 5)

    sampling_start_iter = env_int("PD_SAMPLING_START", 1_000)
    sampling_interval = env_int("PD_SAMPLING_INTERVAL", 500)
    min_observable_samples = env_int("PD_MIN_OBS_SAMPLES", 5)
    observable_upsampling_factor = env_int("PD_OBS_FACTOR", 1)

    diagnostic_every = env_int("PD_DIAGNOSTIC_EVERY", 100)
    diagnostic_hist_bins = env_int("PD_DIAGNOSTIC_HIST_BINS", 80)
    diagnostic_sample_size = env_int("PD_DIAGNOSTIC_SAMPLE_SIZE", 100_000)

    init_type = os.environ.get("PD_INIT_TYPE", "hard_field")
    use_wandb = env_bool("PD_USE_WANDB", True)
    save_diagnostic_plots = env_bool("PD_SAVE_DIAGNOSTIC_PLOTS", True)

    output_root = Path(
        os.environ.get(
            "PD_OUTPUT_DIR",
            str(PROJECT_ROOT / "results" / "k4_disassortative_d12_diagnostics"),
        )
    )

    run_name = (
        f"K4_disassortative_d{d}_H{H}"
        f"_m{mparisi:g}_M{M}_seed{seed}"
    )

    print(f"Array/config index: {config_index}", flush=True)
    print(f"Configuration {config_index + 1}/{len(CONFIGS)}: {cfg}", flush=True)
    print(f"Project root: {PROJECT_ROOT}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"Compiled CUDA: {torch.version.cuda}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    pd = PopDyn(
        K=K,
        d=d,
        H=H,
        problem_type="disassortative",
        mparisi=mparisi,
        M=M,
        damping=damping,
        max_iter=max_iter,
        tol=tol,
        convergence_check_every=convergence_check_every,
        track_diff=True,
        diff_n_bins=60,
        eps=1e-300,
        num_samples=num_samples,
        observable_upsampling_factor=observable_upsampling_factor,
        observable_batch_size=observable_batch_size,
        min_observable_samples=min_observable_samples,
        sampling_start_iter=sampling_start_iter,
        sampling_interval=sampling_interval,
        require_convergence_for_sampling=True,
        init_type=init_type,
        impose_color_symmetry=True,
        seed=seed,
        dtype=np.float64,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compile_core=False,
        use_wandb=use_wandb,
        wandb_project="disassortative-popdyn",
        wandb_group="K4_d12_diagnostic_trials",
        wandb_name=run_name,
        wandb_config_extra={
            "config_index": config_index,
            "trial_label": cfg["label"],
            "purpose": "population-collapse diagnostics",
        },
        log_every=200,
        wandb_log_plots=True,
        save_final_plots=True,
        final_plot_bins=100,
        save_locally=True,
        save_dir=str(output_root),
        run_name=run_name,
        diagnostic_every=diagnostic_every,
        diagnostic_hist_bins=diagnostic_hist_bins,
        diagnostic_sample_size=diagnostic_sample_size,
        save_diagnostic_plots=save_diagnostic_plots,
    )

    pd.run(
        max_iter=max_iter,
        check_convergence=True,
        sample_observables=True,
        num_samples=num_samples,
        reset_samples=True,
        stable_checks_required=stable_checks_required,
        verbose=2,
        finish_wandb=True,
    )

    health = pd.population_health()

    summary: Dict[str, Any] = {
        "config_index": config_index,
        "K": K,
        "d": d,
        "H": H,
        "problem_type": "disassortative",
        "mparisi": mparisi,
        "M": M,
        "seed": seed,
        "init_type": init_type,
        "iteration": int(pd.iteration),
        "last_diff": None if pd.last_diff is None else float(pd.last_diff),
        "num_observable_samples": len(pd.psi_samples),
        "psi_mean": pd.psi_mean,
        "phi_mean": pd.phi_mean,
        "complexity_mean": pd.complexity_mean,
        "s_mean": pd.s_mean,
        "rho_mean": None if pd.rho_mean is None else pd.rho_mean.tolist(),
        "population_min": health["population_min"],
        "population_max": health["population_max"],
        "mean_message_sum": health["mean_message_sum"],
        "mass_min": health["mass_min"],
        "mass_max": health["mass_max"],
        "mass_mean": health["mass_mean"],
        "mass_std": health["mass_std"],
        "max_normalization_error": health["max_normalization_error"],
        "zero_message_fraction": health["zero_message_fraction"],
        "zero_component_fraction": health["zero_component_fraction"],
        "nonfinite_message_fraction": health["nonfinite_message_fraction"],
        "num_diagnostic_snapshots": len(pd.population_diagnostic_history),
        "runtime_sec": pd.diagnostics.get("runtime_sec"),
        "stable": pd.diagnostics.get("stable"),
    }

    print(pd, flush=True)
    print("Final population health:", flush=True)
    for key in (
        "population_min",
        "population_max",
        "mass_min",
        "mass_max",
        "mass_mean",
        "mass_std",
        "max_normalization_error",
        "zero_message_fraction",
        "zero_component_fraction",
        "nonfinite_message_fraction",
    ):
        print(f"  {key}: {summary[key]}", flush=True)

    write_summary(summary, output_root / "summaries")


if __name__ == "__main__":
    main()
