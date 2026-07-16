import numpy as np

from population_dynamics_optimized import PopDyn

pd = PopDyn(
    K=2,
    d=8,
    H=4,
    problem_type="assortative",
    mparisi=1.0,
    M=1_000_000,
    damping=0.8,
    init_type="hard_field",
    device="cuda",
    dtype=np.float64,
    seed=42,

    use_wandb=True,
    wandb_project="population-dynamics",
    wandb_group="K2-assortative",
    wandb_name="K2_d8_H4_m1_M1M",

    wandb_log_plots=True,
    save_final_plots=True,
    final_plot_bins=100,

    save_locally=True,
    save_dir="results/pop_dyn",
)

pd.run(verbose=2)