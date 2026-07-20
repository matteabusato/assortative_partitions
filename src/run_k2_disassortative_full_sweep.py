#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PopDyn import PopDyn


CONFIGS: List[Dict[str, Any]] = [
    # # d = 3
    # {"d": 3, "H": 2, "expected_phase": "easy_RS"},
    # {"d": 3, "H": 3, "expected_phase": "easy_RS"},

    # # d = 4
    # {"d": 4, "H": 3, "expected_phase": "easy_RS"},
    # {"d": 4, "H": 4, "expected_phase": "easy_RS"},

    # # d = 5
    # {"d": 5, "H": 3, "expected_phase": "easy_RS"},
    # {"d": 5, "H": 4, "expected_phase": "easy_RS"},
    # {"d": 5, "H": 5, "expected_phase": "easy_RS"},

    # # d = 6
    # {"d": 6, "H": 4, "expected_phase": "easy_RS"},
    # {"d": 6, "H": 5, "expected_phase": "easy_RS"},
    # {"d": 6, "H": 6, "expected_phase": "easy_RS"},

    # # d = 7
    # {"d": 7, "H": 4, "expected_phase": "easy_RS"},
    # {"d": 7, "H": 5, "expected_phase": "easy_RS"},
    # {"d": 7, "H": 6, "expected_phase": "easy_RS"},
    # {"d": 7, "H": 7, "expected_phase": "easy_RS"},

    # # d = 8
    # {"d": 8, "H": 4, "expected_phase": "frozen_1RSB"},
    # {"d": 8, "H": 5, "expected_phase": "easy_RS"},
    # {"d": 8, "H": 6, "expected_phase": "easy_RS"},
    # {"d": 8, "H": 7, "expected_phase": "easy_RS"},
    # {"d": 8, "H": 8, "expected_phase": "easy_RS"},

    # # d = 9
    # {"d": 9, "H": 5, "expected_phase": "easy_RS"},
    # {"d": 9, "H": 6, "expected_phase": "easy_RS"},
    # {"d": 9, "H": 7, "expected_phase": "easy_RS"},
    # {"d": 9, "H": 8, "expected_phase": "easy_RS"},
    # {"d": 9, "H": 9, "expected_phase": "easy_RS"},

    # # d = 10
    # {"d": 10, "H": 5, "expected_phase": "frozen_1RSB"},
    # {"d": 10, "H": 6, "expected_phase": "easy_RS"},
    # {"d": 10, "H": 7, "expected_phase": "easy_RS"},
    # {"d": 10, "H": 8, "expected_phase": "easy_RS"},
    # {"d": 10, "H": 9, "expected_phase": "easy_RS"},
    # {"d": 10, "H": 10, "expected_phase": "easy_RS"},

    # # d = 11
    # {"d": 11, "H": 6, "expected_phase": "easy_RS"},
    # {"d": 11, "H": 7, "expected_phase": "easy_RS"},
    # {"d": 11, "H": 8, "expected_phase": "easy_RS"},
    # {"d": 11, "H": 9, "expected_phase": "easy_RS"},
    # {"d": 11, "H": 10, "expected_phase": "easy_RS"},
    # {"d": 11, "H": 11, "expected_phase": "easy_RS"},

    # # d = 12
    # {"d": 12, "H": 6, "expected_phase": "frozen_1RSB"},
    # {"d": 12, "H": 7, "expected_phase": "easy_RS"},
    # {"d": 12, "H": 8, "expected_phase": "easy_RS"},
    # {"d": 12, "H": 9, "expected_phase": "easy_RS"},
    # {"d": 12, "H": 10, "expected_phase": "easy_RS"},
    # {"d": 12, "H": 11, "expected_phase": "easy_RS"},

    # # d = 13
    # {"d": 13, "H": 6, "expected_phase": "unclear"},
    # {"d": 13, "H": 7, "expected_phase": "easy_RS"},
    # {"d": 13, "H": 8, "expected_phase": "easy_RS"},
    # {"d": 13, "H": 9, "expected_phase": "easy_RS"},
    # {"d": 13, "H": 10, "expected_phase": "easy_RS"},
    # {"d": 13, "H": 11, "expected_phase": "easy_RS"},

    # # d = 14
    # {"d": 14, "H": 7, "expected_phase": "easy_RS"},
    # {"d": 14, "H": 8, "expected_phase": "easy_RS"},
    # {"d": 14, "H": 9, "expected_phase": "easy_RS"},
    # {"d": 14, "H": 10, "expected_phase": "easy_RS"},
    # {"d": 14, "H": 11, "expected_phase": "easy_RS"},

    # # d = 15
    # {"d": 15, "H": 7, "expected_phase": "easy_RS"},
    # {"d": 15, "H": 8, "expected_phase": "easy_RS"},
    # {"d": 15, "H": 9, "expected_phase": "easy_RS"},
    # {"d": 15, "H": 10, "expected_phase": "easy_RS"},
    # {"d": 15, "H": 11, "expected_phase": "easy_RS"},

    # # d = 16
    # {"d": 16, "H": 7, "expected_phase": "easy_RS"},
    # {"d": 16, "H": 8, "expected_phase": "easy_RS"},
    # {"d": 16, "H": 9, "expected_phase": "easy_RS"},
    # {"d": 16, "H": 10, "expected_phase": "easy_RS"},
    # {"d": 16, "H": 11, "expected_phase": "easy_RS"},

    # # d = 17
    # {"d": 17, "H": 7, "expected_phase": "easy_RS"},
    # {"d": 17, "H": 8, "expected_phase": "easy_RS"},
    # {"d": 17, "H": 9, "expected_phase": "easy_RS"},
    # {"d": 17, "H": 10, "expected_phase": "easy_RS"},
    # {"d": 17, "H": 11, "expected_phase": "easy_RS"},

    # d = 18
    {"d": 18, "H": 9, "expected_phase": "frozen_1RSB"},
    {"d": 18, "H": 10, "expected_phase": "easy_RS"},
    {"d": 18, "H": 11, "expected_phase": "easy_RS"},

    # d = 19
    {"d": 19, "H": 10, "expected_phase": "easy_RS"},
    {"d": 19, "H": 11, "expected_phase": "easy_RS"},

    # d = 20
    {"d": 20, "H": 10, "expected_phase": "frozen_1RSB"},
    {"d": 20, "H": 11, "expected_phase": "easy_RS"},
]


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_optional_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw.strip().lower() in {"", "none", "null"}:
        return None
    return int(raw)


def select_config() -> tuple[int, Dict[str, Any]]:
    raw_index = os.environ.get(
        "SLURM_ARRAY_TASK_ID",
        os.environ.get("PD_CONFIG_INDEX", "0"),
    )
    index = int(raw_index)

    if index < 0 or index >= len(CONFIGS):
        raise IndexError(
            f"Configuration index {index} is outside [0, {len(CONFIGS) - 1}]."
        )

    return index, CONFIGS[index]


def write_summary(summary: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"d{summary['d']}_H{summary['H']}_seed{summary['seed']}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    flat: Dict[str, Any] = {}
    for key, value in summary.items():
        if isinstance(value, list):
            for i, item in enumerate(value):
                flat[f"{key}_{i}"] = item
        elif not isinstance(value, dict):
            flat[key] = value

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)

    print(f"Saved {json_path}", flush=True)
    print(f"Saved {csv_path}", flush=True)


def main() -> None:
    config_index, cfg = select_config()

    K = 2
    d = int(cfg["d"])
    H = int(cfg["H"])
    expected_phase = str(cfg["expected_phase"])

    seed = env_int("PD_SEED", 42)
    M = env_int("PD_M", 1_000_000)
    max_iter = env_int("PD_MAX_ITER", 20_000)
    mparisi = env_float("PD_MPARISI", 1.0)
    damping = env_float("PD_DAMPING", 0.8)
    tol = env_float("PD_TOL", 0.015)

    convergence_check_every = env_int("PD_CONVERGENCE_CHECK_EVERY", 200)
    stable_checks_required = env_int("PD_STABLE_CHECKS", 5)

    observable_upsampling_factor = env_int("PD_OBS_FACTOR", 10)
    observable_batch_size = env_int("PD_OBSERVABLE_BATCH_SIZE", 20_000)
    num_samples = env_optional_int("PD_NUM_SAMPLES")
    min_observable_samples = env_int("PD_MIN_OBS_SAMPLES", 20)
    sampling_start_iter = env_int("PD_SAMPLING_START", 5_000)
    sampling_interval = env_int("PD_SAMPLING_INTERVAL", 500)
    require_convergence_for_sampling = env_bool(
        "PD_REQUIRE_CONVERGENCE_FOR_SAMPLING",
        True,
    )

    diagnostic_every = env_optional_int("PD_DIAGNOSTIC_EVERY")
    diagnostic_hist_bins = env_int("PD_DIAGNOSTIC_HIST_BINS", 80)
    diagnostic_sample_size = env_int("PD_DIAGNOSTIC_SAMPLE_SIZE", 100_000)
    save_diagnostic_plots = env_bool("PD_SAVE_DIAGNOSTIC_PLOTS", False)

    init_type = os.environ.get("PD_INIT_TYPE", "hard_field")
    use_wandb = env_bool("PD_USE_WANDB", True)

    output_root = Path(
        os.environ.get(
            "PD_OUTPUT_DIR",
            str(PROJECT_ROOT / "results" / "k2_disassortative_full_sweep"),
        )
    )

    run_name = (
        f"K2_disassortative_d{d}_H{H}_{expected_phase}"
        f"_m{mparisi:g}_M{M}_seed{seed}"
    )

    print(f"Array task: {config_index}", flush=True)
    print(
        f"Running configuration {config_index + 1}/{len(CONFIGS)}: {cfg}",
        flush=True,
    )
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
        require_convergence_for_sampling=require_convergence_for_sampling,
        init_type=init_type,
        impose_color_symmetry=True,
        seed=seed,
        dtype=np.float64,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compile_core=False,
        use_wandb=use_wandb,
        wandb_project="disassortative-popdyn",
        wandb_group="K2_disassortative_full_sweep",
        wandb_name=run_name,
        wandb_config_extra={
            "config_index": config_index,
            "expected_phase": expected_phase,
            "sweep_name": "K2_disassortative_full_sweep",
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

    mean_message = pd.population.mean(dim=0).detach().cpu().numpy()
    std_message = pd.population.std(dim=0).detach().cpu().numpy()

    summary: Dict[str, Any] = {
        "config_index": config_index,
        "K": K,
        "d": d,
        "H": H,
        "problem_type": "disassortative",
        "expected_phase": expected_phase,
        "mparisi": mparisi,
        "M": M,
        "seed": seed,
        "init_type": init_type,
        "iteration": int(pd.iteration),
        "last_diff": None if pd.last_diff is None else float(pd.last_diff),
        "num_observable_samples": len(pd.psi_samples),
        "psi_mean": pd.psi_mean,
        "psi_std": pd.psi_std,
        "phi_mean": pd.phi_mean,
        "phi_std": pd.phi_std,
        "complexity_mean": pd.complexity_mean,
        "complexity_std": pd.complexity_std,
        "s_mean": pd.s_mean,
        "s_std": pd.s_std,
        "rho_mean": None if pd.rho_mean is None else pd.rho_mean.tolist(),
        "rho_std": None if pd.rho_std is None else pd.rho_std.tolist(),
        "mean_message": mean_message.tolist(),
        "std_message": std_message.tolist(),
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
        "stabilized_iteration": pd.diagnostics.get("stabilized_iteration"),
        "observable_node_samples": pd.diagnostics.get(
            "last_observable_node_samples"
        ),
        "observable_edge_samples": pd.diagnostics.get(
            "last_observable_edge_samples"
        ),
        "observable_batch_size": pd.diagnostics.get("observable_batch_size"),
    }

    print(pd, flush=True)
    print("Mean message:", flush=True)
    print(mean_message, flush=True)
    print("Std message:", flush=True)
    print(std_message, flush=True)
    print(
        f"Population min/max: "
        f"{health['population_min']:.9g}, {health['population_max']:.9g}",
        flush=True,
    )
    print(
        f"Maximum message-normalization error: "
        f"{health['max_normalization_error']:.3e}",
        flush=True,
    )

    write_summary(summary, output_root / "summaries")


if __name__ == "__main__":
    main()