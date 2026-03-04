import numpy as np
from scipy.optimize import least_squares
import time
import json
import math

from decimal import Decimal, getcontext

getcontext().prec = 80
getcontext().Emax = 999999
getcontext().Emin = -999999

D0 = Decimal(0)
D1 = Decimal(1)
D2 = Decimal(2)

def toD(x) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))

def absD(x: Decimal) -> Decimal:
    return x.copy_abs()

def expD(x: Decimal) -> Decimal:
    return x.exp()

def lnD(x: Decimal) -> Decimal:
    return x.ln()

def powD(base: Decimal, exp: int) -> Decimal:
    if exp == 0:
        return D1
    if base == D0:
        return D0
    return base ** exp

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def wandb_sanitize(x):
    if isinstance(x, Decimal):
        return float(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)

    if isinstance(x, np.ndarray):
        return [wandb_sanitize(v) for v in x.tolist()]

    if isinstance(x, dict):
        return {k: wandb_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [wandb_sanitize(v) for v in x]
    
    return x

def json_decimal(obj):
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

class Factorial():
    def __init__(self, D, H):
        self.D = D
        self.H = H

        factorials = np.zeros((D-H+2, D-H+2))
        for i in range(D-H+2):
            r = i+H-1
            c1 = math.comb(D, r)
            for j in range(D-H+2):
                k = j
                c2 = math.comb(D-r, k)
                factorials[i][j] = Decimal(c1 * c2)

        self.factorials = factorials                  

    def get_factorial_chi(self, r, k):
        return toD(self.factorials[r-H+1][k])*toD(self.D-r-k)/toD(self.D)
    
    def get_factorial_Z_node(self, r, k):
        return self.factorials[r-H+1][k]

def assign_f(i):
    if i == 0:
        return [0, 1, 2]
    elif i == 1:
        return [1, 2, 0]
    else:
        return [2, 0, 1]
    
def normalize_chi_decimal(chi):
    total = D0
    for i in range(3):
        for j in range(3):
            total += chi[i, j]

    if total != D0:
        for i in range(3):
            for j in range(3):
                chi[i, j] = chi[i, j] / total
    return chi

def find_current_mu(D, M_star, chi, mu0=np.zeros(3), loss='linear', settingmu="previous"):
    D_int = int(D) if isinstance(D, (int, np.integer)) else int(float(D))
    Df = float(D_int)
    M_star_f = np.array([float(M_star[i]) for i in range(3)], dtype=float)
    chi_f = np.array([[float(chi[i, j]) for j in range(3)] for i in range(3)], dtype=float)
    if isinstance(mu0, np.ndarray) and mu0.dtype == object:
        mu0_f = np.array([float(mu0[i]) for i in range(3)], dtype=float)
    else:
        mu0_f = np.array(mu0, dtype=float)

    if settingmu == "zero":
        mu0_f = np.zeros(3, dtype=float)
    elif settingmu == "previous":
        mu0_f = mu0_f

    def m_values_f(mu):
        mu1, mu2, mu3 = mu

        den = (
            np.exp((2.0 / Df) * mu1) * (chi_f[0, 0] ** 2)
            + np.exp((2.0 / Df) * mu2) * (chi_f[1, 1] ** 2)
            + np.exp((2.0 / Df) * mu3) * (chi_f[2, 2] ** 2)
            + 2.0 * (
                np.exp((1.0 / Df) * (mu1 + mu2)) * chi_f[0, 1] * chi_f[1, 0]
                + np.exp((1.0 / Df) * (mu2 + mu3)) * chi_f[1, 2] * chi_f[2, 1]
                + np.exp((1.0 / Df) * (mu3 + mu1)) * chi_f[2, 0] * chi_f[0, 2]
            )
        )

        return np.array([
            (
                np.exp((2.0 / Df) * mu1) * (chi_f[0, 0] ** 2)
                + np.exp((1.0 / Df) * (mu1 + mu2)) * (chi_f[0, 1] * chi_f[1, 0])
                + np.exp((1.0 / Df) * (mu1 + mu3)) * (chi_f[0, 2] * chi_f[2, 0])
            ) / den,
            (
                np.exp((2.0 / Df) * mu2) * (chi_f[1, 1] ** 2)
                + np.exp((1.0 / Df) * (mu2 + mu1)) * (chi_f[1, 0] * chi_f[0, 1])
                + np.exp((1.0 / Df) * (mu2 + mu3)) * (chi_f[1, 2] * chi_f[2, 1])
            ) / den,
            (
                np.exp((2.0 / Df) * mu3) * (chi_f[2, 2] ** 2)
                + np.exp((1.0 / Df) * (mu3 + mu1)) * (chi_f[2, 0] * chi_f[0, 2])
                + np.exp((1.0 / Df) * (mu3 + mu2)) * (chi_f[2, 1] * chi_f[1, 2])
            ) / den,
        ], dtype=float)

    def residuals_f(mu):
        return m_values_f(mu) - M_star_f

    res = least_squares(
        residuals_f,
        mu0_f,
        method="trf",
        loss=loss,
        xtol=2.23e-16,
        ftol=1e-21,
        gtol=1e-21,
    )

    mu_sol_f = -res.x # -!!

    mu_sol = np.array([Decimal(str(mu_sol_f[i])) for i in range(3)], dtype=object)
    return mu_sol

def update_chi(D: int, H: int, M, THRESHOLD, MAX_ITER, chi, damping: Decimal, mu0, settingmu, loss, FACTORIALS):
    Dd = toD(D)

    chi_new = chi.copy()

    if settingmu != "always_zero":
        mu = find_current_mu(D, M, chi, mu0, loss=loss, settingmu=settingmu)
    else:
        mu = np.array([D0, D0, D0], dtype=object)

    for i in range(3):
        f = assign_f(i)

        second_term = D0
        for r in range(H, D):
            for k in range(0, D - r):
                term = powD(chi[f[0], i], r) * powD(chi[f[1], i], k) * powD(chi[f[2], i], D - 1 - r - k)
                second_term += toD(FACTORIALS.get_factorial_chi(r,k)) * term

        chi_new[i, f[1]] = damping * expD(-(D1 / Dd) * (mu[i] + mu[f[1]])) * second_term + (D1 - damping) * chi[i, f[1]]
        chi_new[i, f[2]] = damping * expD(-(D1 / Dd) * (mu[i] + mu[f[2]])) * second_term + (D1 - damping) * chi[i, f[2]]

        for k in range(0, D - H - 1):
            term = powD(chi[f[0], i], r) * powD(chi[f[1], i], k) * powD(chi[f[2], i], D - 1 - r - k)
            second_term += toD(FACTORIALS.get_factorial_chi(r,k)) * term

        chi_new[i, f[0]] = damping * expD(-(D1 / Dd) * (mu[i] + mu[f[0]])) * second_term + (D1 - damping) * chi[i, f[0]]

    chi_new = normalize_chi_decimal(chi_new)
    return chi_new, mu

def compute_Z_node(D: int, H: int, chi, FACTORIALS):
    Z_node = D0
    for i in range(3):
        f = assign_f(i)
        for r in range(H, D + 1):
            for k in range(0, D - r + 1):
                term = powD(chi[f[0], i], r) * powD(chi[f[1], i], k) * powD(chi[f[2], i], D - r - k)
                Z_node += toD(FACTORIALS.get_factorial_Z_node(r, k)) * term
    return Z_node

def compute_Z_edge(D: int, mu, chi):
    Dd = toD(D)
    Z_edge = D0

    for i in range(3):
        Z_edge += expD((D2 / Dd) * mu[i]) * (chi[i, i] ** 2)

        j = (i + 1) % 3
        Z_edge += D2 * expD((D1 / Dd) * (mu[i] + mu[j])) * chi[i, j] * chi[j, i]

    return Z_edge

def compute_phi_RS(D: int, Z_node: Decimal, Z_edge: Decimal):
    Dd = toD(D)
    return lnD(Z_node) - (Dd / D2) * lnD(Z_edge)

def compute_m_actual(D: int, mu, Z_edge: Decimal, chi):
    Dd = toD(D)
    m_actual = np.array([D0, D0, D0], dtype=object)

    m_actual[0] = (
        expD((D2 / Dd) * mu[0]) * (chi[0, 0] ** 2)
        + expD((D1 / Dd) * (mu[0] + mu[1])) * chi[0, 1] * chi[1, 0]
        + expD((D1 / Dd) * (mu[0] + mu[2])) * chi[0, 2] * chi[2, 0]
    ) / Z_edge

    m_actual[1] = (
        expD((D2 / Dd) * mu[1]) * (chi[1, 1] ** 2)
        + expD((D1 / Dd) * (mu[1] + mu[0])) * chi[1, 0] * chi[0, 1]
        + expD((D1 / Dd) * (mu[1] + mu[2])) * chi[1, 2] * chi[2, 1]
    ) / Z_edge

    m_actual[2] = (
        expD((D2 / Dd) * mu[2]) * (chi[2, 2] ** 2)
        + expD((D1 / Dd) * (mu[2] + mu[0])) * chi[2, 0] * chi[0, 2]
        + expD((D1 / Dd) * (mu[2] + mu[1])) * chi[2, 1] * chi[1, 2]
    ) / Z_edge

    return m_actual

def compute_entropy(phi_RS: Decimal, mu, m_actual):
    dot = D0
    for i in range(3):
        dot += mu[i] * m_actual[i]
    return phi_RS + dot

def compute_quantities(D: int, H: int, chi, mu, FACTORIALS):
    Z_node = compute_Z_node(D, H, chi, FACTORIALS)
    Z_edge = compute_Z_edge(D, mu, chi)
    phi_RS = compute_phi_RS(D, Z_node, Z_edge)
    m_actual = compute_m_actual(D, mu, Z_edge, chi)
    s = compute_entropy(phi_RS, mu, m_actual)
    return Z_node, Z_edge, phi_RS, m_actual, s

def chi_metrics_decimal(chi_new, chi_old):
    diffs = []
    vals = []
    for i in range(3):
        for j in range(3):
            d = chi_new[i, j] - chi_old[i, j]
            diffs.append(absD(d))
            vals.append(chi_new[i, j])

    diff_max = max(diffs)

    # entropy: -sum(p ln p) for p>0
    ent = Decimal(0)
    for p in vals:
        if p > 0:
            ent -= p * p.ln()

    return {
        "chi_diff_max": diff_max,
        "chi_entropy": ent,
    }

def run_bp(D, H, M, THRESHOLD, MAX_ITER, chi, damping, mu0, settingmu, log_every=1000, use_wandb=False, loss='linear', FACTORIALS=None):
    mu = mu0.copy()
    iter = 0
    t0 = time.time()

    last_log_t = t0

    while iter < MAX_ITER:
        chi_old = chi.copy()
        chi_new, mu = update_chi(D, H, M, THRESHOLD, MAX_ITER, chi, damping, mu0, settingmu, loss, FACTORIALS)
        
        metrics = chi_metrics_decimal(chi_new, chi_old)
        diff = metrics["chi_diff_max"]
        chi = chi_new.copy()

        if iter%log_every == 0:
            elapsed = time.time() - t0
            step_dt = time.time() - last_log_t
            last_log_t = time.time()

            Z_node_tmp, Z_edge_tmp, phi_RS_tmp, m_actual_tmp, s_tmp = compute_quantities(D, H, chi, mu, FACTORIALS)
            m_err = m_actual_tmp - M

            log_payload = {
                "iter": iter,
                "elapsed_sec": elapsed,
                "sec_since_last_log": step_dt,
                **metrics,
                "chi00": chi[0, 0],
                "chi01": chi[0, 1],
                "chi02": chi[0, 2],
                "chi10": chi[1, 0],
                "chi11": chi[1, 1],
                "chi12": chi[1, 2],
                "chi20": chi[2, 0],
                "chi21": chi[2, 1],
                "chi22": chi[2, 2],
                "mu_0": mu[0],
                "mu_1": mu[1],
                "mu_2": mu[2],
                "m_actual_0": m_actual_tmp[0],
                "m_actual_1": m_actual_tmp[1],
                "m_actual_2": m_actual_tmp[2],
                "total_m_actual": np.sum(m_actual_tmp),
                "Z_node": Z_node_tmp,
                "Z_edge": Z_edge_tmp,
                "phi_RS": phi_RS_tmp,
                "chi": [chi[i, j] for i in range(3) for j in range(3)],
                "s": s_tmp,
            }

            if use_wandb and WANDB_AVAILABLE:
                wandb.log(wandb_sanitize(log_payload), step=iter)

        if diff < THRESHOLD:
            break
        iter += 1

    total_time = time.time() - t0
    converged = (diff < THRESHOLD)
    return chi, mu, iter, total_time, converged

if __name__ == "__main__":
    K = 3  # Number of groups
    N = 1002
    D = 20

    if (N*D) % 2 != 0 or (N % K != 0):
        raise ValueError("N*D must be even to construct a valid graph without self-loops or multiple edges and N must be a multiple of K.")

    H = 6
    M = np.array([D1/3, D1/3, D1/3], dtype=object)
    THRESHOLD = toD("1e-21")
    MAX_ITER = 10000000
    LOG_EVERY = 1000
    N_RUNS = 1
    DAMPING = toD("0.01")
    MU0 = np.array([D0, D0, D0], dtype=object)
    settingmu = "previous"  # can be "always_zero", "zero", "previous",
    loss_mu = "soft_l1"  # can be "linear", "soft_l1", "huber", "cauchy", "arctan"
    initialization_chi = "personalized"  # can be "uniform", "unif_diag", "one_hot", "gaussian", "one_hot_softmax"

    FACTORIALS = Factorial(D, H)

    # print(FACTORIALS.factorials)

    # for r in range(H-1, D ):
    #     for k in range(0, D - r):
    #         print( FACTORIALS.get_factorial_chi(r, k))

    for _ in range(N_RUNS):
        SEED = np.random.randint(0, 1000000)
        np.random.seed(SEED)
        epsilon = toD("50e-2")

        if initialization_chi == "uniform":
            chi = np.array([[D1, D1, D1],[D1, D1, D1],[D1, D1, D1]], dtype=object) # dimension (K,K)
        elif initialization_chi == "unif_diag":
            chi = np.zeros((3,3), dtype=float)
            chi[0,0] = 1/3
            chi[1,1] = 1/3
            chi[2,2] = 1/3
        elif initialization_chi == "one_hot":
            chi = np.zeros((3,3), dtype=float)
            chi[0,0] = 1.0
        elif initialization_chi == "one_hot_softmax":
            chi = np.zeros((3,3), dtype=float)
            chi[0,0] = 1.0 - 8*1e-2
            chi[0,1] = 1e-2
            chi[0,2] = 1e-2
            chi[1,0] = 1e-2
            chi[1,1] = 1e-2
            chi[1,2] = 1e-2
            chi[2,0] = 1e-2
            chi[2,1] = 1e-2
            chi[2,2] = 1e-2
        elif initialization_chi == "personalized":
            chi = np.empty((3,3), dtype=object)
            for i in range(3):
                for j in range(3):
                    chi[i,j] = D0
            chi[0,0] = D1 - epsilon
            chi[1,1] = epsilon/D2
            chi[2,2] = epsilon/D2
            # chi = np.zeros((3,3), dtype=float)
            # chi[0,0] = 1 - epsilon
            # chi[1,1] = epsilon/2
            # chi[2,2] = epsilon/2
        else:  # "gaussian"
            chi = np.random.rand(3, 3)

        chi = normalize_chi_decimal(chi)

        USE_WANDB = True

        if USE_WANDB and not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed in this environment. `pip install wandb` first.")

        if USE_WANDB:
            wandb.init(
                project="bp_fixed_point",
                name=f"N{N}_D{D}_H{H}_seed{SEED}_{initialization_chi}_{settingmu}_{epsilon}_chi_dec",
                group=f"FIXED_N{N}_D{D}_{settingmu}", 
                config={
                    "N": N,
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
                json.dump({"chi_init": chi.tolist()}, f, indent=2, default=json_decimal)
            wandb.save("chi_init.json")    

        chi, mu, iters, total_time, converged = run_bp(D, H, M, THRESHOLD, MAX_ITER, chi, DAMPING, MU0, settingmu, LOG_EVERY,
            USE_WANDB, loss_mu, FACTORIALS)

        Z_node, Z_edge, phi_RS, m_actual, s = compute_quantities(D, H, chi, mu, FACTORIALS)
        # n_solutions = float(np.exp(s * N))

        if USE_WANDB:
            wandb.summary["converged"] = bool(converged)
            wandb.summary["iters"] = int(iters)
            wandb.summary["total_time_sec"] = float(total_time)
            wandb.summary["Z_node"] = float(Z_node)
            wandb.summary["Z_edge"] = float(Z_edge)
            wandb.summary["phi_RS"] = float(phi_RS)
            wandb.summary["s"] = float(s)
            # wandb.summary["n_solutions_est"] = float(n_solutions)
            wandb.summary["m_err_l2_final"] = float(np.linalg.norm(m_actual - M))
            wandb.summary["m_err_maxabs_final"] = float(np.max(np.abs(m_actual - M)))

            with open("final_results.json", "w") as f:
                json.dump({
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
                }, f, indent=2, default=json_decimal)
            wandb.save("final_results.json")
            wandb.finish()
