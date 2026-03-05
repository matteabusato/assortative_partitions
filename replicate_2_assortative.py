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

        factorials = np.empty((D - H + 2,), dtype=object)
        for i in range(D-H+2):
            r = i+H-1
            factorials[i] = Decimal(math.comb(D, r))

        self.factorials = factorials                  

    def get_factorial_chi(self, r):
        return toD(self.factorials[r-H+1])*toD(self.D-r)/toD(self.D)
    
    def get_factorial_Z_node(self, r):
        return self.factorials[r-H+1]

def assign_f(i):
    if i == 0:
        return [0, 1]
    elif i == 1:
        return [1, 0]
    
def normalize_chi_decimal(chi):
    total = D0
    for i in range(2):
        for j in range(2):
            total += chi[i, j]

    if total != D0:
        for i in range(2):
            for j in range(2):
                chi[i, j] = chi[i, j] / total
    return chi


## fatto da qua in su 

def find_current_mu(D, M_star, chi, mu0=np.zeros(3), loss='linear', settingmu="previous"):
    D_int = int(D) if isinstance(D, (int, np.integer)) else int(float(D))
    Df = float(D_int)
    M_star_f = np.array([float(M_star[i]) for i in range(2)], dtype=float)
    chi_f = np.array([[float(chi[i, j]) for j in range(2)] for i in range(2)], dtype=float)
    if isinstance(mu0, np.ndarray) and mu0.dtype == object:
        mu0_f = np.array([float(mu0[i]) for i in range(2)], dtype=float)
    else:
        mu0_f = np.array(mu0, dtype=float)

    if settingmu == "zero":
        mu0_f = np.zeros(2, dtype=float)
    elif settingmu == "previous":
        mu0_f = mu0_f

    def m_values_f(mu):
        mu1, mu2 = mu

        den = ( np.exp((2.0 / Df) * mu1) * (chi_f[0, 0] ** 2) 
               + np.exp((2.0 / Df) * mu2) * (chi_f[1, 1] ** 2) 
               + 2 * np.exp((1.0 / Df) * (mu1 + mu2)) * chi_f[1, 0] * chi_f[0, 1]
        )

        return np.array([
            (   
                np.exp((2.0 / Df) * mu1) * (chi_f[0, 0] ** 2) 
                + np.exp((1.0 / Df) * (mu1 + mu2)) * chi_f[1, 0] * chi_f[0, 1]
            ) / den,
            (
                np.exp((2.0 / Df) * mu2) * (chi_f[1, 1] ** 2)
                + np.exp((1.0 / Df) * (mu2 + mu1)) * (chi_f[1, 0] * chi_f[0, 1])
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

    mu_sol = np.array([Decimal(str(mu_sol_f[i])) for i in range(2)], dtype=object)
    return mu_sol

def update_chi(D: int, H: int, M, THRESHOLD, MAX_ITER, chi, damping: Decimal, mu0, settingmu, loss, FACTORIALS):
    Dd = toD(D)

    chi_new = chi.copy()

    if settingmu != "always_zero":
        mu = find_current_mu(D, M, chi, mu0, loss=loss, settingmu=settingmu)
    else:
        mu = np.array([D0, D0], dtype=object)

    for i in range(2):
        f = assign_f(i)

        second_term = D0
        for r in range(H, D):
            term = powD(chi[f[0], i], r) * powD(chi[f[1], i], D-1-r)
            second_term += toD(FACTORIALS.get_factorial_chi(r)) * term

        chi_new[i, f[1]] = damping * expD(-(D1 / Dd) * (mu[i] + mu[f[1]])) * second_term + (D1 - damping) * chi[i, f[1]]

        r = H-1
        term = powD(chi[f[0], i], r) * powD(chi[f[1], i], D-1-r)
        second_term += toD(FACTORIALS.get_factorial_chi(r)) * term

        chi_new[i, f[0]] = damping * expD(-(D1 / Dd) * (mu[i] + mu[f[0]])) * second_term + (D1 - damping) * chi[i, f[0]]

    chi_new = normalize_chi_decimal(chi_new)
    return chi_new, mu

def compute_Z_node(D: int, H: int, chi, FACTORIALS):
    Z_node = D0
    for i in range(2):
        f = assign_f(i)
        for r in range(H, D + 1):
            term = powD(chi[f[0], i], r) * powD(chi[f[1], i], D-r)
            Z_node += toD(FACTORIALS.get_factorial_Z_node(r)) * term
    return Z_node

def compute_Z_edge(D: int, mu, chi):
    Dd = toD(D)
    Z_edge = D0

    for i in range(2):
        Z_edge += expD((D2 / Dd) * mu[i]) * (chi[i, i] ** 2)

        j = (i + 1) % 2
        Z_edge += expD((D1 / Dd) * (mu[i] + mu[j])) * chi[i, j] * chi[j, i]

    return Z_edge

def compute_phi_RS(D: int, Z_node: Decimal, Z_edge: Decimal):
    Dd = toD(D)
    return lnD(Z_node) - (Dd / D2) * lnD(Z_edge)

def compute_m_actual(D: int, mu, Z_edge: Decimal, chi):
    Dd = toD(D)
    m_actual = np.array([D0, D0], dtype=object)

    m_actual[0] = (
        expD((D2 / Dd) * mu[0]) * (chi[0, 0] ** 2)
        + expD((D1 / Dd) * (mu[0] + mu[1])) * chi[0, 1] * chi[1, 0]
    ) / Z_edge

    m_actual[1] = (
        expD((D2 / Dd) * mu[1]) * (chi[1, 1] ** 2)
        + expD((D1 / Dd) * (mu[1] + mu[0])) * chi[1, 0] * chi[0, 1]
    ) / Z_edge

    return m_actual

def compute_entropy(phi_RS: Decimal, mu, m_actual):
    dot = D0
    for i in range(2):
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
    for i in range(2):
        for j in range(2):
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
                wandb.log(wandb_sanitize(log_payload), step=iter)

        if diff < THRESHOLD:
            break
        iter += 1

    total_time = time.time() - t0
    converged = (diff < THRESHOLD)
    return chi, mu, iter, total_time, converged

if __name__ == "__main__":
    D = 7
    Hs = [5]

    magnetizations = [np.array([toD(EPS), D1-toD(EPS)], dtype = object) for EPS in np.arange(0.31,1, 0.1)]
    THRESHOLD = toD("1e-21")
    MAX_ITER = 10000000
    DAMPING = toD("0.01")
    MU0 = np.array([D0, D0], dtype=object)
    LOG_EVERY = 1000
    settingmu = "previous"  # can be "always_zero", "zero", "previous",
    loss_mu = "soft_l1"  # can be "linear", "soft_l1", "huber", "cauchy", "arctan"
    initialization_chi = "uniform"  # can be "uniform", "unif_diag", "one_hot", "gaussian", "one_hot_softmax"

    for H in Hs:
        FACTORIALS = Factorial(D, H)
        for i, M in enumerate(magnetizations):
            SEED = np.random.randint(0, 1000000)
            np.random.seed(SEED)
            
            chi = np.array([[D1, D1],[D1, D1]], dtype=object) # dimension (K,K)
            chi = normalize_chi_decimal(chi)

            USE_WANDB = True

            if USE_WANDB and not WANDB_AVAILABLE:
                raise ImportError("wandb is not installed in this environment. `pip install wandb` first.")

            if USE_WANDB:
                wandb.init(
                    project="bp_fixed_point",
                    name=f"D{D}_H{H}_{i}_prova1",
                    group=f"REPLICATE_2ASSORTATIVE__D{D}", 
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
                    json.dump({"chi_init": chi.tolist()}, f, indent=2, default=json_decimal)
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
                wandb.summary["s_final"] = float(s)

                with open("final_results.json", "w") as f:
                    json.dump({
                        "chi_final": chi.tolist(),
                        "mu_final": mu.tolist(),
                        "m_actual": m_actual.tolist(),
                        "M_target": M.tolist(),
                        "Z_node": Z_node,
                        "Z_edge": Z_edge,
                        "phi_RS": phi_RS,
                        "s_final": s,
                        "iters": int(iters),
                        "total_time_sec": float(total_time),
                        "converged": bool(converged),
                    }, f, indent=2, default=json_decimal)
                wandb.save("final_results.json")
                wandb.finish()
