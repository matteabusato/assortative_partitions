from __future__ import annotations

import numpy as np

from bp import BPConfig, BeliefPropagation
from init import ChiInitSpec, make_chi_init
from logging_utils import (
    wandb_init_if_enabled,
    wandb_set_summary_if_enabled,
    wandb_finish_if_enabled,
    save_json,
    wandb_save_if_enabled,
)


def main():
    N = 1002
    D = 100
    H = 32
    M = np.array([1/3, 1/3, 1/3], dtype=float)

    THRESHOLD = 1e-21
    MAX_ITER = 10_000_000
    LOG_EVERY = 1000
    DAMPING = 0.01

    settingmu = "previous"      # "always_zero", "zero", "previous"
    loss_mu = "soft_l1"         # "linear", "soft_l1", "huber", "cauchy", "arctan"

    chi_spec = ChiInitSpec(mode="unif_diag", epsilon=0.01, softmax_eps=1e-2)

    USE_WANDB = True
    N_RUNS = 1

    if (N * D) % 2 != 0:
        raise ValueError("N*D must be even.")
    if N % 3 != 0:
        raise ValueError("N must be a multiple of K=3.")

    for _ in range(N_RUNS):
        SEED = int(np.random.randint(0, 1_000_000))
        chi_init = make_chi_init(chi_spec, seed=SEED)

        cfg = BPConfig(
            N=N,
            D=D,
            H=H,
            M_target=M,
            threshold=THRESHOLD,
            max_iter=MAX_ITER,
            log_every=LOG_EVERY,
            damping=DAMPING,
            settingmu=settingmu,
            loss_mu=loss_mu,
            use_wandb=USE_WANDB,
        )

        run_name = f"N{N}_D{D}_H{H}_seed{SEED}_{chi_spec.mode}_{settingmu}"
        group = f"NEW_N{N}_D{D}_{settingmu}"

        wandb_init_if_enabled(
            use_wandb=USE_WANDB,
            project="bp_fixed_point",
            name=run_name,
            group=group,
            config={
                "seed": SEED,
                "chi_init_mode": chi_spec.mode,
                "chi_init_epsilon": chi_spec.epsilon,
                "chi_init_softmax_eps": chi_spec.softmax_eps,
                "M_target": M.tolist(),
                **cfg.__dict__,
            },
        )

        # Save initial chi
        save_json("chi_init.json", {"chi_init": chi_init})
        wandb_save_if_enabled(USE_WANDB, "chi_init.json")

        bp = BeliefPropagation(cfg, chi_init=chi_init, mu_init=np.zeros(3))
        out = bp.run()

        save_json(
            "final_results.json",
            {
                "seed": SEED,
                "chi_final": out["chi"],
                "mu_final": out["mu"],
                "m_actual": out["m_actual"],
                "M_target": M,
                "Z_node": out["Z_node"],
                "Z_edge": out["Z_edge"],
                "phi_RS": out["phi_RS"],
                "s": out["s"],
                "iters": out["iters"],
                "total_time_sec": out["total_time_sec"],
                "converged": out["converged"],
            },
        )
        wandb_save_if_enabled(USE_WANDB, "final_results.json")

        wandb_set_summary_if_enabled(
            USE_WANDB,
            {
                "converged": bool(out["converged"]),
                "iters": int(out["iters"]),
                "total_time_sec": float(out["total_time_sec"]),
                "Z_node": float(out["Z_node"]),
                "Z_edge": float(out["Z_edge"]),
                "phi_RS": float(out["phi_RS"]),
                "s": float(out["s"]),
                "m_err_l2_final": float(np.linalg.norm(out["m_actual"] - M)),
                "m_err_maxabs_final": float(np.max(np.abs(out["m_actual"] - M))),
            },
        )

        wandb_finish_if_enabled(USE_WANDB)


if __name__ == "__main__":
    main()