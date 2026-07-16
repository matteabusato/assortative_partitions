from __future__ import annotations

import csv
import os
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

from population_dynamics_optimized import PopDyn


CONFIGS = [
    {"d": 8, "H": 4, "expected_s0": 0.36259, "phase": "easy_RS"},
    {"d": 8, "H": 5, "expected_s0": 0.02302, "phase": "frozen_1RSB"},
    {"d": 10, "H": 5, "expected_s0": 0.34723, "phase": "easy_RS"},
    {"d": 10, "H": 6, "expected_s0": 0.04004, "phase": "frozen_1RSB"},
]


def log_final_plots(pd: PopDyn, out_dir: str) -> None:
    """Reproduce and log the same final plots used by the NumPy sweep."""
    os.makedirs(out_dir, exist_ok=True)

    plot_specs = [
        (
            "population_histograms",
            "population_histograms.png",
            lambda path: pd.plot_population_histograms(
                n_bins=200,
                density=False,
                save_path=path,
                show=False,
            ),
        ),
        (
            "population_atoms",
            "population_atoms.png",
            lambda path: pd.plot_population_atoms(
                save_path=path,
                show=False,
            ),
        ),
        (
            "mean_message",
            "mean_message.png",
            lambda path: pd.plot_mean_message(
                save_path=path,
                show=False,
            ),
        ),
    ]

    if len(pd.psi_samples) > 0:
        plot_specs.append(
            (
                "observable_samples",
                "observable_samples.png",
                lambda path: pd.plot_observable_samples(
                    save_path=path,
                    show=False,
                ),
            )
        )

    paths: Dict[str, str] = {}
    for key, filename, plotter in plot_specs:
        path = os.path.join(out_dir, filename)
        fig = plotter(path)
        paths[key] = path
        plt.close(fig)

    if pd.use_wandb and pd.wandb_run is not None and wandb is not None:
        pd.wandb_run.log(
            {f"plots/{key}": wandb.Image(path) for key, path in paths.items()},
            step=pd.iteration,
        )


def summarize_run(pd: PopDyn, cfg: Dict[str, Any]) -> Dict[str, Any]:
    expected = float(cfg["expected_s0"])

    if cfg["phase"] == "easy_RS":
        retrieved = pd.phi_mean
        retrieval_name = "phi_mean"
    else:
        retrieved = pd.complexity_mean
        retrieval_name = "complexity_mean"

    return {
        "d": int(cfg["d"]),
        "H": int(cfg["H"]),
        "phase": str(cfg["phase"]),
        "expected_s0": expected,
        "retrieval_name": retrieval_name,
        "retrieved": retrieved,
        "abs_error": abs(retrieved - expected) if retrieved is not None else np.nan,
        "psi_mean": pd.psi_mean,
        "psi_std": pd.psi_std,
        "phi_mean": pd.phi_mean,
        "phi_std": pd.phi_std,
        "complexity_mean": pd.complexity_mean,
        "complexity_std": pd.complexity_std,
        "s_mean": pd.s_mean,
        "s_std": pd.s_std,
        "rho0_mean": pd.rho_mean[0] if pd.rho_mean is not None else np.nan,
        "rho1_mean": pd.rho_mean[1] if pd.rho_mean is not None else np.nan,
        "rho0_std": pd.rho_std[0] if pd.rho_std is not None else np.nan,
        "rho1_std": pd.rho_std[1] if pd.rho_std is not None else np.nan,
        "last_diff": pd.last_diff,
        "finished_iter": pd.iteration,
        "runtime_sec": pd.diagnostics.get("runtime_sec", np.nan),
        "num_observable_samples": len(pd.psi_samples),
        "stable": pd.diagnostics.get("stable", None),
        "device": str(pd.device),
        "dtype": str(pd.dtype),
    }


def print_population_diagnostics(pd: PopDyn) -> None:
    """Compute compact diagnostics on GPU and copy only small results to CPU."""
    with torch.inference_mode():
        mean_message = pd.population.mean(dim=0).detach().cpu().numpy()
        std_message = pd.population.std(dim=0, unbiased=False).detach().cpu().numpy()
        min_value = float(pd.population.min().item())
        max_value = float(pd.population.max().item())
        sums_ok = bool(
            torch.allclose(
                pd.population.sum(dim=(1, 2)),
                torch.ones(pd.M, dtype=pd.torch_dtype, device=pd.device),
                rtol=1e-6,
                atol=1e-8,
            )
        )

    print(pd)
    print("mean message:")
    print(mean_message)
    print("std message:")
    print(std_message)
    print("min/max:", min_value, max_value)
    print("matrix sums:", sums_ok)


def _wandb_summary_payload(summary: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in summary.items():
        if isinstance(value, (bool, int, float, np.integer, np.floating)):
            if isinstance(value, np.generic):
                value = value.item()
            payload[f"summary/{key}"] = value
    return payload


def run_one(cfg: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    d = int(cfg["d"])
    H = int(cfg["H"])
    phase = str(cfg["phase"])

    wandb_name = f"K2_d{d}_H{H}_{phase}_mparisi1_M1mill_seed{seed}"
    out_dir = f"results/k2_table2/d{d}_H{H}_seed{seed}"

    pd = PopDyn(
        K=2,
        d=d,
        H=H,
        problem_type="assortative",
        mparisi=1.0,
        M=1_000_000,
        damping=0.8,
        max_iter=20_000,
        tol=0.015,
        convergence_check_every=200,
        diff_n_bins=200,
        sampling_start_iter=5_000,
        sampling_interval=500,
        min_observable_samples=20,
        observable_upsampling_factor=100,
        require_convergence_for_sampling=True,
        init_type="hard_field",
        impose_color_symmetry=True,
        seed=seed,
        device="cuda",
        dtype=np.float64,
        use_wandb=True,
        wandb_project="assortative-popdyn",
        wandb_group="K2_table2_reproduction_optimized",
        wandb_name=wandb_name,
        log_every=200,
        wandb_config_extra={
            "paper": "Behrens_Arpino_Kivva_Zdeborova_2022",
            "target_table": "Table 2",
            "expected_s0": cfg["expected_s0"],
            "expected_phase": phase,
            "implementation": "pytorch_cuda_optimized",
        },
        # The sweep reproduces the old explicit plotting logic below.
        wandb_log_plots=False,
        save_final_plots=False,
        save_locally=True,
        save_dir="results/k2_table2",
        run_name=wandb_name,
    )

    try:
        # Keep wandb open because the sweep logs plots and summary afterward.
        pd.run(
            verbose=2,
            stable_checks_required=5,
            finish_wandb=False,
        )

        print_population_diagnostics(pd)
        log_final_plots(pd, out_dir)

        summary = summarize_run(pd, cfg)

        if pd.use_wandb and pd.wandb_run is not None:
            pd.wandb_run.log(_wandb_summary_payload(summary), step=pd.iteration)
            pd.wandb_run.summary.update(summary)

        return summary

    finally:
        if pd.use_wandb and pd.wandb_run is not None:
            pd.wandb_run.finish()
            pd.wandb_run = None


def main() -> None:
    os.makedirs("results/k2_table2", exist_ok=True)

    rows = [run_one(cfg, seed=42) for cfg in CONFIGS]

    out_csv = "results/k2_table2/summary.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSummary")
    for row in rows:
        print(row)

    print(f"\nSaved summary to {out_csv}")


if __name__ == "__main__":
    main()
