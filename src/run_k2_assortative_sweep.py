import os
import csv
import numpy as np
import matplotlib.pyplot as plt

try:
    import wandb
except ImportError:
    wandb = None

from population_dynamics_new import PopDyn


CONFIGS = [
    {"d": 8, "H": 4, "expected_s0": 0.36259, "phase": "easy_RS"},
    {"d": 8, "H": 5, "expected_s0": 0.02302, "phase": "frozen_1RSB"},
    {"d": 10, "H": 5, "expected_s0": 0.34723, "phase": "easy_RS"},
    {"d": 10, "H": 6, "expected_s0": 0.04004, "phase": "frozen_1RSB"},
]


def log_final_plots(pd: PopDyn, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    paths = {}

    fig = pd.plot_population_histograms(
        n_bins=200,
        density=False,
        save_path=os.path.join(out_dir, "population_histograms.png"),
        show=False,
    )
    paths["population_histograms"] = os.path.join(out_dir, "population_histograms.png")
    plt.close(fig)

    fig = pd.plot_population_atoms(
        save_path=os.path.join(out_dir, "population_atoms.png"),
        show=False,
    )
    paths["population_atoms"] = os.path.join(out_dir, "population_atoms.png")
    plt.close(fig)

    fig = pd.plot_mean_message(
        save_path=os.path.join(out_dir, "mean_message.png"),
        show=False,
    )
    paths["mean_message"] = os.path.join(out_dir, "mean_message.png")
    plt.close(fig)

    if len(pd.psi_samples) > 0:
        fig = pd.plot_observable_samples(
            save_path=os.path.join(out_dir, "observable_samples.png"),
            show=False,
        )
        paths["observable_samples"] = os.path.join(out_dir, "observable_samples.png")
        plt.close(fig)

    if pd.use_wandb and pd.wandb_run is not None and wandb is not None:
        data = {}
        for key, path in paths.items():
            data[f"plots/{key}"] = wandb.Image(path)
        pd.wandb_run.log(data, step=pd.iteration)


def summarize_run(pd: PopDyn, cfg: dict) -> dict:
    expected = cfg["expected_s0"]

    if cfg["phase"] == "easy_RS":
        retrieved = pd.phi_mean
        retrieval_name = "phi_mean"
    elif cfg["phase"] == "frozen_1RSB":
        retrieved = pd.complexity_mean
        retrieval_name = "complexity_mean"
    else:
        retrieved = pd.complexity_mean
        retrieval_name = "complexity_mean"

    return {
        "d": cfg["d"],
        "H": cfg["H"],
        "phase": cfg["phase"],
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
        "num_observable_samples": len(pd.psi_samples),
        "stable": pd.diagnostics.get("stable", None),
    }


def run_one(cfg: dict, seed: int = 42) -> dict:
    d = cfg["d"]
    H = cfg["H"]
    phase = cfg["phase"]

    wandb_name = f"K2_d{d}_H{H}_{phase}_mparisi1_M1mill_seed{seed}"

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

        use_wandb=True,
        wandb_project="assortative-popdyn",
        wandb_group="K2_table2_reproduction",
        wandb_name=wandb_name,
        log_every=200,
        wandb_config_extra={
            "paper": "Behrens_Arpino_Kivva_Zdeborova_2022",
            "target_table": "Table 2",
            "expected_s0": cfg["expected_s0"],
            "expected_phase": phase,
        },
    )

    pd.run(verbose=2, stable_checks_required=5)

    print(pd)
    print("mean message:")
    print(pd.population.mean(axis=0))
    print("std message:")
    print(pd.population.std(axis=0))
    print("min/max:", pd.population.min(), pd.population.max())
    print("matrix sums:", np.allclose(pd.population.sum(axis=(1, 2)), 1.0))

    out_dir = f"results/k2_table2/d{d}_H{H}_seed{seed}"
    log_final_plots(pd, out_dir)

    summary = summarize_run(pd, cfg)

    if pd.use_wandb and pd.wandb_run is not None:
        pd.wandb_run.log({f"summary/{k}": v for k, v in summary.items() if isinstance(v, (int, float, np.floating, bool))})
        pd.wandb_run.finish()

    return summary


def main() -> None:
    os.makedirs("results/k2_table2", exist_ok=True)

    rows = []

    for cfg in CONFIGS:
        rows.append(run_one(cfg, seed=42))

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