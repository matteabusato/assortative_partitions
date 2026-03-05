import numpy as np
from scipy.optimize import least_squares
import time
import json
import math

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class Factorial():
    def __init__(self, D, H):
        self.D = D
        self.H = H

        factorials = np.empty((D - H + 2,), dtype=object)
        for i in range(D-H+2):
            r = i+H-1
            factorials[i] = math.comb(D, r)

        self.factorials = factorials                  

    def get_factorial_chi(self, r):
        return self.factorials[r-H+1] * (self.D-r) / self.D
    
    def get_factorial_Z_node(self, r):
        return self.factorials[r-H+1]

def assign_f(i):
    if i == 0:
        return [0, 1]
    elif i == 1:
        return [1, 0]
    
def normalize_chi(chi):
    chi_sum = np.sum(chi)
    if chi_sum > 1e-300:
        chi = chi / chi_sum
    return chi

def initialize_chi(initialization_chi='uniform', epsilon=0.5):
        chi = np.zeros((2, 2), dtype=float)
        if initialization_chi == "uniform":
            chi = np.ones((2, 2), dtype=float)
        elif initialization_chi == "unif_diag":
            chi[0,0] = 1/2
            chi[1,1] = 1/2
        elif initialization_chi == "one_hot":
            chi[0,0] = 1.0
        elif initialization_chi == "one_hot_softmax":
            chi[0,0] = 1.0 - epsilon
            chi[0,1] = epsilon / 3
            chi[1,0] = epsilon / 3
            chi[1,1] = epsilon / 3
        elif initialization_chi == "personalized":
            chi[0,0] = 1.0 - epsilon
            chi[1,1] = epsilon
        elif "gaussian":
            chi = np.random.rand(2, 2)

        return normalize_chi(chi)

def find_current_mu(D, M_star, chi, mu0=np.zeros(2), loss='linear', settingmu='previous'):
    if settingmu == "zero":
        mu0 = np.zeros(2, dtype=float)
    elif settingmu == "previous":
        mu0 = mu0

    def m_values(mu):
        mu1, mu2 = mu

        den = ( np.exp((2.0 / D) * mu1) * (chi[0, 0] ** 2) 
               + np.exp((2.0 / D) * mu2) * (chi[1, 1] ** 2) 
               + 2 * np.exp((1.0 / D) * (mu1 + mu2)) * chi[1, 0] * chi[0, 1]
        )

        return np.array([
            (   
                np.exp((2.0 / D) * mu1) * (chi[0, 0] ** 2) 
                + np.exp((1.0 / D) * (mu1 + mu2)) * chi[1, 0] * chi[0, 1]
            ) / den,
            (
                np.exp((2.0 / D) * mu2) * (chi[1, 1] ** 2)
                + np.exp((1.0 / D) * (mu2 + mu1)) * (chi[1, 0] * chi[0, 1])
            ) / den,
        ], dtype=float)

    def residuals(mu):
        return m_values(mu) - M_star

    res = least_squares(
        residuals,
        mu0,
        method="trf",
        loss=loss,
        xtol=2.23e-16,
        ftol=1e-21,
        gtol=1e-21,
    )

    return -res.x # -!!

def update_chi(D: int, H: int, M, THRESHOLD, MAX_ITER, chi, damping, mu0, settingmu, loss, FACTORIALS):
    chi_new = chi.copy()

    if settingmu != "always_zero":
        mu = find_current_mu(D, M, chi, mu0, loss=loss, settingmu=settingmu)
    else:
        mu = np.array([0, 0], dtype=float)

    for i in range(2):
        f = assign_f(i)

        second_term = 0.0
        for r in range(H, D):
            term = (chi[f[0], i] ** r) * (chi[f[1], i] ** (D-1-r))
            second_term += FACTORIALS.get_factorial_chi(r) * term

        chi_new[i, f[1]] = damping * np.exp(-(1.0 / D) * (mu[i] + mu[f[1]])) * second_term + (1.0 - damping) * chi[i, f[1]]

        r = H-1
        term = (chi[f[0], i] ** r) * (chi[f[1], i] ** (D-1-r))
        second_term += FACTORIALS.get_factorial_chi(r) * term

        chi_new[i, f[0]] = damping * np.exp(-(1.0 / D) * (mu[i] + mu[f[0]])) * second_term + (1.0 - damping) * chi[i, f[0]]

    chi_new = normalize_chi(chi_new)
    return chi_new, mu

def compute_Z_node(D: int, H: int, chi, FACTORIALS):
    Z_node = 0.0
    for i in range(2):
        f = assign_f(i)
        for r in range(H, D + 1):
            term = (chi[f[0], i] ** r) * (chi[f[1], i] ** (D-r))
            Z_node += FACTORIALS.get_factorial_Z_node(r) * term
    return Z_node

def compute_Z_edge(D: int, mu, chi):
    Z_edge = 0.0

    for i in range(2):
        Z_edge += np.exp((2.0 / D) * mu[i]) * (chi[i, i] ** 2)

        j = (i + 1) % 2
        Z_edge += np.exp((1.0 / D) * (mu[i] + mu[j])) * chi[i, j] * chi[j, i]

    return Z_edge

def compute_phi_RS(D: int, Z_node, Z_edge):
    return np.log(Z_node) - (D / 2.0)*np.log(Z_edge)

def compute_m_actual(D: int, mu, Z_edge, chi):
    m_actual = np.array([0, 0], dtype=float)

    m_actual[0] = (
        np.exp((2.0 / D) * mu[0]) * (chi[0, 0] ** 2)
        + np.exp((1.0 / D) * (mu[0] + mu[1])) * chi[0, 1] * chi[1, 0]
    ) / Z_edge

    m_actual[1] = (
        np.exp((2.0 / D) * mu[1]) * (chi[1, 1] ** 2)
        + np.exp((1.0 / D) * (mu[1] + mu[0])) * chi[1, 0] * chi[0, 1]
    ) / Z_edge

    return m_actual

def compute_entropy(phi_RS, mu, m_actual):
    return phi_RS + np.dot(mu, m_actual)

def compute_quantities(D: int, H: int, chi, mu, FACTORIALS):
    Z_node = compute_Z_node(D, H, chi, FACTORIALS)
    Z_edge = compute_Z_edge(D, mu, chi)
    phi_RS = compute_phi_RS(D, Z_node, Z_edge)
    m_actual = compute_m_actual(D, mu, Z_edge, chi)
    s = compute_entropy(phi_RS, mu, m_actual)
    return Z_node, Z_edge, phi_RS, m_actual, s

def chi_metrics(chi_new, chi_old):
    diff = chi_new - chi_old
    return {
        "chi_diff_max": float(np.max(np.abs(diff))),
        "chi_diff_mean_abs": float(np.mean(np.abs(diff))),
    }

def run_bp(D, H, M, THRESHOLD, MAX_ITER, chi, damping, mu0, settingmu, log_every=1000, use_wandb=False, loss='linear', FACTORIALS=None):
    mu = mu0.copy()
    iter = 0
    t0 = time.time()

    last_log_t = t0

    while iter < MAX_ITER:
        chi_old = chi.copy()
        chi_new, mu = update_chi(D, H, M, THRESHOLD, MAX_ITER, chi, damping, mu0, settingmu, loss, FACTORIALS)
        
        metrics = chi_metrics(chi_new, chi_old)
        diff = metrics["chi_diff_max"]
        chi = chi_new.copy()

        if iter%log_every == 0:
            elapsed = time.time() - t0
            step_dt = time.time() - last_log_t
            last_log_t = time.time()

            Z_node_tmp, Z_edge_tmp, phi_RS_tmp, m_actual_tmp, s_tmp = compute_quantities(D, H, chi, mu, FACTORIALS)

            log_payload = {
                "iter": iter,
                "elapsed_sec": elapsed,
                "sec_since_last_log": step_dt,
                **metrics,
                "chi00": chi[0, 0],
                "chi01": chi[0, 1],
                "chi10": chi[1, 0],
                "chi11": chi[1, 1],
                "mu_0": mu[0],
                "mu_1": mu[1],
                "m_actual_0": m_actual_tmp[0],
                "m_actual_1": m_actual_tmp[1],
                "total_m_actual": np.sum(m_actual_tmp),
                "Z_node": Z_node_tmp,
                "Z_edge": Z_edge_tmp,
                "phi_RS": phi_RS_tmp,
                "chi": [chi[i, j] for i in range(2) for j in range(2)],
                "s": s_tmp,
            }

            if use_wandb and WANDB_AVAILABLE:
                wandb.log(log_payload, step=iter)

        if diff < THRESHOLD:
            break
        iter += 1

    total_time = time.time() - t0
    converged = (diff < THRESHOLD)
    return chi, mu, iter, total_time, converged

if __name__ == "__main__":
    Ds = [7]
    Hs = [1]
    Ms = {np.array([1/2, 1/2], dtype=float)}
    THRESHOLD = 1e-21
    MAX_ITER = 10000000
    LOG_EVERY = 1000
    DAMPING = 0.01
    MU0 = np.zeros(2)
    settingmu = "previous"  # can be "always_zero", "zero", "previous",
    loss_mu = "soft_l1"  # can be "linear", "soft_l1", "huber", "cauchy", "arctan"
    initialization_chi = "uniform"  # can be "uniform", "unif_diag", "one_hot", "gaussian", "one_hot_softmax"
    epsilon = 0.5

    for D in Ds:
        for H in Hs:
            FACTORIALS = Factorial(D, H)
            for M in Ms:
                SEED = np.random.randint(0, 1000000)
                np.random.seed(SEED)

                chi = initialize_chi(initialization_chi, epsilon)

                USE_WANDB = True

                if USE_WANDB and not WANDB_AVAILABLE:
                    raise ImportError("wandb is not installed in this environment. `pip install wandb` first.")

                if USE_WANDB:
                    wandb.init(
                        project="bp_fixed_point",
                        name=f"D{D}_H{H}_{initialization_chi}_{settingmu}",
                        group=f"FINAL_2ASS_D{D}_{settingmu}", 
                        config={
                            "D": D,
                            "H": H,
                            "M_target": M.tolist(),
                            "THRESHOLD": THRESHOLD,
                            "MAX_ITER": MAX_ITER,
                            "damping": DAMPING,
                            "mu0": MU0.tolist(),
                            "settingmu": settingmu,
                            "LOG_EVERY": LOG_EVERY,
                            "chi_init": chi.tolist(),
                            "initialization_chi": initialization_chi,
                            "seed": SEED,
                            "loss_mu": loss_mu,
                        },
                    )

                    with open("chi_init.json", "w") as f:
                        json.dump({"chi_init": chi.tolist()}, f, indent=2)
                    wandb.save("chi_init.json")    

                chi, mu, iters, total_time, converged = run_bp(D, H, M, THRESHOLD, MAX_ITER, chi, DAMPING, MU0, settingmu, LOG_EVERY,
                    USE_WANDB, loss_mu, FACTORIALS)

                Z_node, Z_edge, phi_RS, m_actual, s = compute_quantities(D, H, chi, mu, FACTORIALS)

                if USE_WANDB:
                    wandb.summary["converged"] = bool(converged)
                    wandb.summary["iters"] = int(iters)
                    wandb.summary["total_time_sec"] = float(total_time)
                    wandb.summary["Z_node"] = float(Z_node)
                    wandb.summary["Z_edge"] = float(Z_edge)
                    wandb.summary["phi_RS"] = float(phi_RS)
                    wandb.summary["s"] = float(s)

                    with open("final_results.json", "w") as f:
                        json.dump({
                            "D": D,
                            "H": H,
                            "chi_final": chi.tolist(),
                            "mu_final": mu.tolist(),
                            "m_actual": m_actual.tolist(),
                            "M_target": M.tolist(),
                            "Z_node": Z_node,
                            "Z_edge": Z_edge,
                            "phi_RS": phi_RS,
                            "s": s,
                            "iters": int(iters),
                            "total_time_sec": float(total_time),
                            "converged": bool(converged),
                        }, f, indent=2)
                        
                    wandb.save("final_results.json")
                    wandb.finish()