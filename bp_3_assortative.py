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

def assign_f(i):
    if i == 0:
        return 0, 1, 2
    elif i == 1:
        return 1, 2, 0
    else:
        return 2, 0, 1
    
def normalize_chi(chi):
    chi_sum = np.sum(chi)
    if chi_sum > 1e-300:
        chi = chi / chi_sum
    return chi

def find_current_mu(D, M_star, chi, mu0=np.zeros(3)):
    def m_values(mu):
        mu1, mu2, mu3 = mu
        return np.array([
            -mu1 * (np.exp((2/D)*mu1)*(chi[0, 0]**2)+ np.exp((1/D)*(mu1+mu2))*(chi[0, 1]*chi[1, 0]) + np.exp((1/D)*(mu1+mu3))*(chi[0, 2]*chi[2, 0])) 
            / (np.exp((2/D)*mu1)*(chi[0,0]**2) + np.exp((2/D)*mu2)*(chi[1,1]**2) + np.exp((2/D)*mu3)*(chi[2,2]**2) + 2*(np.exp((1/D)*(mu1+mu2))*chi[0,1]*chi[1,0]) + 2*(np.exp((1/D)*(mu2+mu3))*chi[1,2]*chi[2,1]) + 2*(np.exp((1/D)*(mu3+mu1)) *chi[2,0]*chi[0,2]) ),
            -mu2 * (np.exp((2/D)*mu2)*(chi[1, 1]**2)+ np.exp((1/D)*(mu2+mu1))*(chi[1, 0]*chi[0, 1]) + np.exp((1/D)*(mu2+mu3))*(chi[1, 2]*chi[2, 1])) 
            / (np.exp((2/D)*mu1)*(chi[0,0]**2) + np.exp((2/D)*mu2)*(chi[1,1]**2) + np.exp((2/D)*mu3)*(chi[2,2]**2) + 2*(np.exp((1/D)*(mu1+mu2))*chi[0,1]*chi[1,0]) + 2*(np.exp((1/D)*(mu2+mu3))*chi[1,2]*chi[2,1]) + 2*(np.exp((1/D)*(mu3+mu1)) *chi[2,0]*chi[0,2]) ),
            -mu3 * (np.exp((2/D)*mu3)*(chi[2, 2]**2)+ np.exp((1/D)*(mu3+mu1))*(chi[2, 0]*chi[0, 2]) + np.exp((1/D)*(mu3+mu2))*(chi[2, 1]*chi[1, 2])) 
            / (np.exp((2/D)*mu1)*(chi[0,0]**2) + np.exp((2/D)*mu2)*(chi[1,1]**2) + np.exp((2/D)*mu3)*(chi[2,2]**2) + 2*(np.exp((1/D)*(mu1+mu2))*chi[0,1]*chi[1,0]) + 2*(np.exp((1/D)*(mu2+mu3))*chi[1,2]*chi[2,1]) + 2*(np.exp((1/D)*(mu3+mu1)) *chi[2,0]*chi[0,2]) ),
        ], dtype=float)
    def residuals(mu):
        return m_values(mu) - M_star 

    mu0 = mu0
    res = least_squares(residuals, mu0, method="trf")

    return res.x

def update_chi(D, H, M, THRESHOLD, MAX_ITER, chi, damping, mu0):
    mu = find_current_mu(D, M, chi, mu0)

    for i in range(3):
        for j in range(3):
            second_term = 0
            f1, f2, f3 = assign_f(i)
            for r in range(H-(i==j), D-1):
                for k in range(D-1-r):
                    second_term += math.comb(D-1, r) * math.comb(D-1-r, k) * (chi[f1, i]**r) * (chi[f2,i]**k) * (chi[f3,i]**(D-1-r-k))

            chi[i, j] = damping * np.exp(- (1/D)*(mu[i]+mu[j])) * second_term + (1-damping) * chi[i, j]

    chi = normalize_chi(chi)
    return chi, mu

def compute_Z_node(D, H, chi):
    Z_node = 0
    for i in range(3):
        f1, f2, f3 = assign_f(i)
        for r in range(H, D):
            for k in range(D-r):
                Z_node += math.comb(D, r) * math.comb(D-r, k) * (chi[f1, i]**r) * (chi[f2,i]**k) * (chi[f3,i]**(D-r-k))
    return Z_node

def compute_Z_edge(D, mu, chi):
    Z_edge = 0
    for i in range(3):
        Z_edge += np.exp((2/D)*mu[i])*(chi[i, i]**2)
        j = (i+1)%3
        Z_edge += 2 * np.exp((1/D)*(mu[i]+mu[j])) * chi[i, j] * chi[j, i]

    return Z_edge

def compute_phi_RS(D, Z_node, Z_edge):
    return np.log(Z_node) - (D/2)*np.log(Z_edge)

def compute_m_actual(D, mu, Z_edge, chi):
    m_actual = np.zeros(3)
    m_actual[0] = - mu[0]* (1/Z_edge) * ( np.exp( (2/D) * mu[0]) * chi[0,0]**2 + np.exp( (1/D) * (mu[0]+mu[1]) ) * chi[0,1]*chi[1,0] + np.exp( (1/D) * (mu[0]+mu[2]) ) * chi[0,2]*chi[2,0] )
    m_actual[1] = - mu[1]* (1/Z_edge) * ( np.exp( (2/D) * mu[1]) * chi[1,1]**2 + np.exp( (1/D) * (mu[1]+mu[0]) ) * chi[1,0]*chi[0,1] + np.exp( (1/D) * (mu[1]+mu[2]) ) * chi[1,2]*chi[2,1] )
    m_actual[2] = - mu[2]* (1/Z_edge) * ( np.exp( (2/D) * mu[2]) * chi[2,2]**2 + np.exp( (1/D) * (mu[2]+mu[0]) ) * chi[2,0]*chi[0,2] + np.exp( (1/D) * (mu[2]+mu[1]) ) * chi[2,1]*chi[1,2] )
    return m_actual

def compute_entropy(phi_RS, mu, m_actual):
    return phi_RS + np.dot(mu, m_actual)

def compute_quantities(D, H, chi, mu):
    Z_node = compute_Z_node(D, H, chi)
    Z_edge = compute_Z_edge(D, mu, chi)
    phi_RS = compute_phi_RS(D, Z_node, Z_edge)
    m_actual = compute_m_actual(D, mu, Z_edge, chi)
    s = compute_entropy(phi_RS, mu, m_actual)
    return Z_node, Z_edge, phi_RS, m_actual, s

def chi_metrics(chi_new, chi_old):
    diff = chi_new - chi_old
    return {
        "chi_diff_max": float(np.max(np.abs(diff))),
        "chi_diff_l2": float(np.linalg.norm(diff)),
        "chi_diff_mean_abs": float(np.mean(np.abs(diff))),
        "chi_min": float(np.min(chi_new)),
        "chi_max": float(np.max(chi_new)),
        "chi_entropy": float(-np.sum(np.where(chi_new > 0, chi_new * np.log(chi_new), 0.0))),
    }

def run_bp(D, H, M, THRESHOLD, MAX_ITER, chi, damping, mu0, log_every=1000, use_wandb=False):
    mu = mu0.copy()
    iter = 0
    t0 = time.time()

    last_log_t = t0

    while iter < MAX_ITER:
        chi_old = chi.copy()
        chi_new, mu = update_chi(D, H, M, THRESHOLD, MAX_ITER, chi, damping, mu)
        
        metrics = chi_metrics(chi_new, chi_old)
        diff = metrics["chi_diff_max"]
        chi = chi_new

        if iter%log_every == 0:
            elapsed = time.time() - t0
            step_dt = time.time() - last_log_t
            last_log_t = time.time()

            Z_node_tmp, Z_edge_tmp, phi_RS_tmp, m_actual_tmp, s_tmp = compute_quantities(D, H, chi, mu)
            m_err = m_actual_tmp - M

            log_payload = {
                "iter": iter,
                "elapsed_sec": elapsed,
                "sec_since_last_log": step_dt,
                **metrics,
                "mu_0": float(mu[0]),
                "mu_1": float(mu[1]),
                "mu_2": float(mu[2]),
                "m_actual_0": float(m_actual_tmp[0]),
                "m_actual_1": float(m_actual_tmp[1]),
                "m_actual_2": float(m_actual_tmp[2]),
                "total_m_actual": float(np.sum(m_actual_tmp)),
                "m_err_l2": float(np.linalg.norm(m_err)),
                "m_err_maxabs": float(np.max(np.abs(m_err))),
                "Z_node": float(Z_node_tmp),
                "Z_edge": float(Z_edge_tmp),
                "phi_RS": float(phi_RS_tmp),
                "s": float(s_tmp),
                "estimated_n_solutions": float(np.exp(s_tmp * N))
            }

            # print(f"Iteration {iter}, max diff in chi: {diff:.10e}")
            # print(f"chi =\n{chi}\n")

            if use_wandb and WANDB_AVAILABLE:
                wandb.log(log_payload, step=iter)

        if diff < THRESHOLD:
            break
        iter += 1

    total_time = time.time() - t0
    converged = (diff < THRESHOLD)
    return chi, mu, iter, total_time, converged

if __name__ == "__main__":
    # K = 3  ->  3 groups in the partition
    N = 15
    D = 7
    H = 3
    M = np.array([1/3, 1/3, 1/3])
    THRESHOLD = 1e-10
    MAX_ITER = 1000000
    LOG_EVERY = 1000
    N_RUNS = 10
    DAMPING = 0.01  
    MU0 = np.zeros(3)

    for _ in range(N_RUNS):
        SEED = np.random.randint(0, 1000000)
        np.random.seed(SEED)

        chi = np.ones((3,3), dtype=float) # dimension (K,K)
        chi /= chi.sum()

        USE_WANDB = True

        if USE_WANDB and not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed in this environment. `pip install wandb` first.")

        if USE_WANDB:
            wandb.init(
                project="bp_fixed_point",
                name=f"N{N}_D{D}_H{H}_damp{DAMPING}_seed{SEED}",
                group=f"N{N}_D{D}_H{H}_M{M}", 
                config={
                    "N": N,
                    "D": D,
                    "H": H,
                    "M_target": M.tolist(),
                    "THRESHOLD": THRESHOLD,
                    "MAX_ITER": MAX_ITER,
                    "damping": DAMPING,
                    "mu0": MU0.tolist(),
                    "LOG_EVERY": LOG_EVERY,
                    "chi_init": chi.tolist(),
                    "seed": SEED,
                },
            )

            with open("chi_init.json", "w") as f:
                json.dump({"chi_init": chi.tolist()}, f, indent=2)
            wandb.save("chi_init.json")

        chi, mu, iters, total_time, converged = run_bp(D, H, M, THRESHOLD, MAX_ITER, chi, DAMPING, MU0, log_every=LOG_EVERY,
            use_wandb=USE_WANDB)

        Z_node, Z_edge, phi_RS, m_actual, s = compute_quantities(D, H, chi, mu)
        n_solutions = float(np.exp(s * N))

        print("Found fixed point chi = \n", chi, " after iters: ", iters, " time(s): ", total_time)
        print("Z_node = ", Z_node, ", Z_edge = ", Z_edge, ", phi_RS = ", phi_RS,
            ", actual magnetization = ", m_actual, ", entropy = ", s)
        print("The number of solution with magnetization close to target ", M, " is ", n_solutions)

        if USE_WANDB:
            wandb.summary["converged"] = bool(converged)
            wandb.summary["iters"] = int(iters)
            wandb.summary["total_time_sec"] = float(total_time)
            wandb.summary["Z_node"] = float(Z_node)
            wandb.summary["Z_edge"] = float(Z_edge)
            wandb.summary["phi_RS"] = float(phi_RS)
            wandb.summary["s"] = float(s)
            wandb.summary["n_solutions_est"] = float(n_solutions)
            wandb.summary["m_err_l2_final"] = float(np.linalg.norm(m_actual - M))
            wandb.summary["m_err_maxabs_final"] = float(np.max(np.abs(m_actual - M)))

            with open("final_results.json", "w") as f:
                json.dump({
                    "chi_final": chi.tolist(),
                    "mu_final": mu.tolist(),
                    "m_actual": m_actual.tolist(),
                    "M_target": M.tolist(),
                    "Z_node": float(Z_node),
                    "Z_edge": float(Z_edge),
                    "phi_RS": float(phi_RS),
                    "s": float(s),
                    "iters": int(iters),
                    "total_time_sec": float(total_time),
                    "converged": bool(converged),
                    "n_solutions_est": float(n_solutions),
                }, f, indent=2)

            wandb.save("final_results.json")
            wandb.finish()