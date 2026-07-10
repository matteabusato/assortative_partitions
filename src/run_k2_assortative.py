import os
import numpy as np
import matplotlib.pyplot as plt
from wandb import config

try:
    import wandb
except ImportError:
    wandb = None

from population_dynamics_new import PopDyn


def log_population_summary(pd: PopDyn) -> None:
    if not pd.use_wandb or pd.wandb_run is None:
        return

    mean_chi = pd.population.mean(axis=0)
    std_chi = pd.population.std(axis=0)

    data = {
        "population/min": float(pd.population.min()),
        "population/max": float(pd.population.max()),
        "population/matrix_sum_error_max": float(
            np.max(np.abs(pd.population.sum(axis=(1, 2)) - 1.0))
        ),
    }

    for x in range(pd.K):
        for y in range(pd.K):
            data[f"chi_mean/{x}{y}"] = float(mean_chi[x, y])
            data[f"chi_std/{x}{y}"] = float(std_chi[x, y])

    pd.wandb_run.log(data, step=pd.iteration)


def log_plots(pd: PopDyn, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    hist_path = os.path.join(out_dir, "population_histograms.png")
    mean_path = os.path.join(out_dir, "mean_message.png")
    atoms_path = os.path.join(out_dir, "population_atoms.png")
    obs_path = os.path.join(out_dir, "observable_samples.png")

    fig = pd.plot_population_histograms(
        n_bins=200,
        density=False,
        save_path=hist_path,
        show=False,
    )
    plt.close(fig)

    fig = pd.plot_mean_message(
        save_path=mean_path,
        show=False,
    )
    plt.close(fig)

    fig = pd.plot_population_atoms(
        save_path=atoms_path,
        show=False,
    )
    plt.close(fig)

    if len(pd.psi_samples) > 0:
        fig = pd.plot_observable_samples(
            save_path=obs_path,
            show=False,
        )
        plt.close(fig)

    if pd.use_wandb and pd.wandb_run is not None and wandb is not None:
        plot_data = {
            "plots/population_histograms": wandb.Image(hist_path),
            "plots/mean_message": wandb.Image(mean_path),
            "plots/population_atoms": wandb.Image(atoms_path),
        }

        if len(pd.psi_samples) > 0:
            plot_data["plots/observable_samples"] = wandb.Image(obs_path)

        pd.wandb_run.log(plot_data, step=pd.iteration)


def main() -> None:
    K=2
    d=8
    H=4
    problem_type="assortative"
    mparisi=1.0
    damping=0.8
    seed=42

    popdyn_config = dict(
        K=K,
        d=d,
        H=H,
        problem_type=problem_type,
        mparisi=mparisi,

        M=1_000_000,
        damping=damping,
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
        wandb_group=f"K{K}_d{d}_H{H}",
        wandb_name=f"K{K}_d{d}_H{H}_m{mparisi}_M1mill_damp{damping}_seed{seed}",
        log_every=200,

        wandb_config_extra={
            "experiment_type": "K2_assortative_test",
            "stable_checks_required": 5,
            "notes": f"K={K}, d={d}, H={H}, hard-field population dynamics",
        },
    )

    pd = PopDyn(**popdyn_config)
    pd.run(verbose=2, stable_checks_required=5)

    pd._reset_observable_samples()

    for _ in range(20):
        pd.update_observables()
        pd._record_current_observables()

    pd._finalize_observable_samples()
    pd.plot_observable_samples()
    print(pd)

    print(pd)
    print("mean message:")
    print(pd.population.mean(axis=0))
    print("std message:")
    print(pd.population.std(axis=0))
    print("min/max:", pd.population.min(), pd.population.max())
    print("matrix sums:", np.allclose(pd.population.sum(axis=(1, 2)), 1.0))

    log_population_summary(pd)
    log_plots(pd, out_dir="results/k2_d8_H5_seed0")

    if pd.use_wandb and pd.wandb_run is not None:
        pd.wandb_run.finish()


if __name__ == "__main__":
    main()