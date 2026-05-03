"""
Tests comparing old factorial-based BP implementations against refactored BP.py.

Covers:
1. K=2 assortative old update_chi vs BP.py update.
2. K=3 assortative old update_chi vs BP.py update.
3. K=3 disassortative old update_chi vs BP.py update.
4. Old Z_node vs BP.py Z_node.
5. Old Z_node vs brute-force enumeration.
6. Old raw message sums vs brute-force enumeration.
7. Edge cases H=0, H=1, H=d, H=d+1 where meaningful.
8. Random chi and mu tests.
9. Damped-normalized one-step update tests.

Important:
- These tests use mu_mode='always_zero' logic manually by passing fixed mu.
- They avoid calling the mu solver, so they isolate only update_chi and Z_node.
"""


from __future__ import annotations

import itertools
import math
import traceback
from dataclasses import dataclass

import numpy as np

import sys
from pathlib import Path

# Folder containing this test file
THIS_DIR = Path(__file__).resolve().parent

# Add this folder to Python's import path
sys.path.insert(0, str(THIS_DIR))

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_CODE_DIR = PROJECT_ROOT / "deprecated"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(OLD_CODE_DIR))

import BP as newbp
import bp_2_assortative as old2ass
import bp_3_assortative as old3ass
import bp_3_disassortative as old3dis


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

ATOL = 1e-12
RTOL = 1e-10


@dataclass
class CaseResult:
    name: str
    passed: bool
    max_abs_err: float = 0.0
    details: str = ""


def normalize(x: np.ndarray) -> np.ndarray:
    s = float(np.sum(x))
    if s <= 1e-300:
        return x.copy()
    return x / s


def random_chi(K: int, rng: np.random.Generator, kind: str) -> np.ndarray:
    if kind == "random":
        chi = rng.random((K, K)) + 0.05
    elif kind == "almost_uniform":
        chi = np.ones((K, K)) + 0.01 * rng.normal(size=(K, K))
        chi = np.maximum(chi, 1e-8)
    elif kind == "diagonal_biased":
        chi = 0.05 * np.ones((K, K))
        np.fill_diagonal(chi, 1.0)
        chi += 0.01 * rng.random((K, K))
    elif kind == "offdiag_biased":
        chi = np.ones((K, K))
        np.fill_diagonal(chi, 0.05)
        chi += 0.01 * rng.random((K, K))
    elif kind == "sparse_soft":
        chi = 1e-6 * np.ones((K, K))
        for a in range(K):
            chi[a, (a + 1) % K] = 1.0 + 0.1 * rng.random()
        chi += 1e-4 * rng.random((K, K))
    else:
        raise ValueError(kind)

    return normalize(chi)


def random_mu(K: int, rng: np.random.Generator) -> np.ndarray:
    mu = rng.normal(size=K)
    mu -= mu.mean()
    return mu


def assert_close(name: str, actual, expected, atol=ATOL, rtol=RTOL) -> CaseResult:
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    err = float(np.max(np.abs(actual - expected))) if actual.size else abs(float(actual - expected))
    passed = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
    details = ""
    if not passed:
        details = (
            f"\nactual =\n{actual}\n"
            f"expected =\n{expected}\n"
            f"diff =\n{actual - expected}\n"
        )
    return CaseResult(name=name, passed=passed, max_abs_err=err, details=details)


# ---------------------------------------------------------------------
# Brute force formulas
# ---------------------------------------------------------------------

def brute_raw_update(
    K: int,
    d: int,
    H: int,
    problem_type: str,
    chi: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    """
    Direct enumeration of all K^(d-1) incoming neighbor-color assignments.

    Returns the unnormalized chi_target, without damping and without final normalization:

        chi_target[a,b]
        =
        exp(-(mu[a]+mu[b])/d)
        sum_{neighbors c_1,...,c_{d-1}}
        1[constraint including removed neighbor b]
        prod_l chi[c_l, a].
    """
    out = np.zeros((K, K), dtype=float)

    for a in range(K):          # root color
        for b in range(K):      # removed / target neighbor color
            total = 0.0
            removed_same = int(a == b)

            for incoming in itertools.product(range(K), repeat=d - 1):
                same_count = removed_same + sum(int(c == a) for c in incoming)

                if problem_type == "assortative":
                    ok = same_count >= H
                elif problem_type == "disassortative":
                    ok = same_count < H
                else:
                    raise ValueError(problem_type)

                if not ok:
                    continue

                weight = 1.0
                for c in incoming:
                    weight *= chi[c, a]
                total += weight

            out[a, b] = math.exp(-(mu[a] + mu[b]) / d) * total

    return out


def brute_Z_node(
    K: int,
    d: int,
    H: int,
    problem_type: str,
    chi: np.ndarray,
) -> float:
    """
    Direct enumeration of all K^d full-neighborhood assignments.
    """
    Z = 0.0

    for a in range(K):
        for neighs in itertools.product(range(K), repeat=d):
            same_count = sum(int(c == a) for c in neighs)

            if problem_type == "assortative":
                ok = same_count >= H
            elif problem_type == "disassortative":
                ok = same_count < H
            else:
                raise ValueError(problem_type)

            if not ok:
                continue

            weight = 1.0
            for c in neighs:
                weight *= chi[c, a]
            Z += weight

    return float(Z)


# ---------------------------------------------------------------------
# Old-code raw update reconstructions
# ---------------------------------------------------------------------

def old2ass_raw_update_from_factorials(
    d: int,
    H: int,
    chi: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    """
    Reconstructs the old K=2 assortative update target BEFORE damping/normalization.
    This mirrors bp_2_assortative.update_chi.
    """
    F = old2ass.Factorial(d, H)
    out = np.zeros_like(chi, dtype=float)

    for i in range(2):
        f = old2ass.assign_f(i)

        S_other = 0.0
        for r in range(H, d):
            term = (chi[f[0], i] ** r) * (chi[f[1], i] ** (d - 1 - r))
            S_other += F.get_factorial_chi(r) * term

        out[i, f[1]] = math.exp(-(mu[i] + mu[f[1]]) / d) * S_other

        S_self = S_other
        r = H - 1
        term = (chi[f[0], i] ** r) * (chi[f[1], i] ** (d - 1 - r))
        S_self += F.get_factorial_chi(r) * term

        out[i, f[0]] = math.exp(-(mu[i] + mu[f[0]]) / d) * S_self

    return out


def old3ass_raw_update_from_factorials(
    d: int,
    H: int,
    chi: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    """
    Reconstructs the old K=3 assortative update target BEFORE damping/normalization.
    This mirrors bp_3_assortative.update_chi.
    """
    F = old3ass.Factorial(d, H)
    out = np.zeros_like(chi, dtype=float)

    for i in range(3):
        f = old3ass.assign_f(i)

        S_other = 0.0
        for r in range(H, d):
            for k in range(0, d - r):
                term = (
                    (chi[f[0], i] ** r)
                    * (chi[f[1], i] ** k)
                    * (chi[f[2], i] ** (d - 1 - r - k))
                )
                S_other += F.get_factorial_chi(r, k) * term

        out[i, f[1]] = math.exp(-(mu[i] + mu[f[1]]) / d) * S_other
        out[i, f[2]] = math.exp(-(mu[i] + mu[f[2]]) / d) * S_other

        S_self = S_other
        r = H - 1
        for k in range(0, d - r):
            term = (
                (chi[f[0], i] ** r)
                * (chi[f[1], i] ** k)
                * (chi[f[2], i] ** (d - 1 - r - k))
            )
            S_self += F.get_factorial_chi(r, k) * term

        out[i, f[0]] = math.exp(-(mu[i] + mu[f[0]]) / d) * S_self

    return out


def old3dis_raw_update_from_factorials(
    d: int,
    H: int,
    chi: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    """
    Reconstructs the old K=3 disassortative update target BEFORE damping/normalization.
    This mirrors bp_3_disassortative.update_chi.
    """
    F = old3dis.Factorial(d, H)
    out = np.zeros_like(chi, dtype=float)

    for i in range(3):
        f = old3dis.assign_f(i)

        S_self = 0.0
        for r in range(0, H - 1):
            for k in range(0, d - r):
                term = (
                    (chi[f[0], i] ** r)
                    * (chi[f[1], i] ** k)
                    * (chi[f[2], i] ** (d - 1 - r - k))
                )
                S_self += F.get_factorial_chi(r, k) * term

        out[i, f[0]] = math.exp(-(mu[i] + mu[f[0]]) / d) * S_self

        S_other = S_self
        r = H - 1
        for k in range(0, d - r):
            term = (
                (chi[f[0], i] ** r)
                * (chi[f[1], i] ** k)
                * (chi[f[2], i] ** (d - 1 - r - k))
            )
            S_other += F.get_factorial_chi(r, k) * term

        out[i, f[1]] = math.exp(-(mu[i] + mu[f[1]]) / d) * S_other
        out[i, f[2]] = math.exp(-(mu[i] + mu[f[2]]) / d) * S_other

    return out


# ---------------------------------------------------------------------
# Old Z_node wrappers
# ---------------------------------------------------------------------

def old_Z_node(K: int, d: int, H: int, problem_type: str, chi: np.ndarray) -> float:
    if K == 2 and problem_type == "assortative":
        F = old2ass.Factorial(d, H)
        return float(old2ass.compute_Z_node(d, H, chi, F))

    if K == 3 and problem_type == "assortative":
        F = old3ass.Factorial(d, H)
        return float(old3ass.compute_Z_node(d, H, chi, F))

    if K == 3 and problem_type == "disassortative":
        F = old3dis.Factorial(d, H)
        return float(old3dis.compute_Z_node(d, H, chi, F))

    raise ValueError((K, problem_type))


def old_raw_update(K: int, d: int, H: int, problem_type: str, chi: np.ndarray, mu: np.ndarray):
    if K == 2 and problem_type == "assortative":
        return old2ass_raw_update_from_factorials(d, H, chi, mu)

    if K == 3 and problem_type == "assortative":
        return old3ass_raw_update_from_factorials(d, H, chi, mu)

    if K == 3 and problem_type == "disassortative":
        return old3dis_raw_update_from_factorials(d, H, chi, mu)

    raise ValueError((K, problem_type))


# ---------------------------------------------------------------------
# One-step damped update comparison against actual old update_chi
# ---------------------------------------------------------------------

def old_actual_one_step(
    K: int,
    d: int,
    H: int,
    problem_type: str,
    chi: np.ndarray,
    mu: np.ndarray,
    damping: float,
) -> np.ndarray:
    """
    Calls the old update_chi with settingmu='always_zero' only when mu is zero.
    For general mu, we manually apply the old raw target + damping + normalization.
    """
    raw = old_raw_update(K, d, H, problem_type, chi, mu)
    return normalize(damping * raw + (1.0 - damping) * chi)


def new_one_step(
    K: int,
    d: int,
    H: int,
    problem_type: str,
    chi: np.ndarray,
    mu: np.ndarray,
    damping: float,
    update_path: str,
) -> np.ndarray:
    if update_path == "fast":
        raw = newbp._update_chi_fast(K, d, H, problem_type, chi, mu)
    elif update_path == "reference":
        raw = newbp._update_chi_reference(K, d, H, problem_type, chi, mu)
    else:
        raise ValueError(update_path)

    return newbp.normalize_chi(damping * raw + (1.0 - damping) * chi)


# ---------------------------------------------------------------------
# Test groups
# ---------------------------------------------------------------------

def run_update_tests(rng: np.random.Generator) -> list[CaseResult]:
    results: list[CaseResult] = []

    configs = [
        # Avoid H=0 for old assortative Factorial, because old code was not written
        # for r=H-1=-1 in the self-message branch.
        (2, "assortative", range(1, 9), range(1, 9)),
        (3, "assortative", range(1, 9), range(1, 9)),
        # Avoid H=0 for old disassortative Factorial, because Factorial has shape (H, D+1).
        (3, "disassortative", range(1, 9), range(1, 9)),
    ]

    chi_kinds = ["random", "almost_uniform", "diagonal_biased", "offdiag_biased", "sparse_soft"]

    for K, problem_type, d_values, _ in configs:
        for d in d_values:
            for H in range(1, d + 1):
                for kind in chi_kinds:
                    for trial in range(5):
                        chi = random_chi(K, rng, kind)
                        mu = random_mu(K, rng)

                        name_base = f"raw update K={K} {problem_type} d={d} H={H} kind={kind} trial={trial}"

                        old_raw = old_raw_update(K, d, H, problem_type, chi, mu)
                        new_ref = newbp._update_chi_reference(K, d, H, problem_type, chi, mu)
                        new_fast = newbp._update_chi_fast(K, d, H, problem_type, chi, mu)
                        brute = brute_raw_update(K, d, H, problem_type, chi, mu)

                        results.append(assert_close(name_base + " | old vs new reference", old_raw, new_ref))
                        results.append(assert_close(name_base + " | old vs new fast", old_raw, new_fast))
                        results.append(assert_close(name_base + " | old vs brute", old_raw, brute, atol=5e-12, rtol=1e-10))

    return results


def run_Z_node_tests(rng: np.random.Generator) -> list[CaseResult]:
    results: list[CaseResult] = []

    configs = [
        (2, "assortative"),
        (3, "assortative"),
        (3, "disassortative"),
    ]

    chi_kinds = ["random", "almost_uniform", "diagonal_biased", "offdiag_biased", "sparse_soft"]

    for K, problem_type in configs:
        for d in range(1, 10):
            # old assortative Factorial supports H=1,...,d+1 in principle
            # old disassortative supports H=1,...,d+1 in principle, but H=d+1 means all configurations pass.
            for H in range(1, d + 2):
                for kind in chi_kinds:
                    for trial in range(5):
                        chi = random_chi(K, rng, kind)

                        name_base = f"Z_node K={K} {problem_type} d={d} H={H} kind={kind} trial={trial}"

                        old_z = old_Z_node(K, d, H, problem_type, chi)
                        new_z = newbp.compute_Z_node(K, d, H, problem_type, chi)
                        brute_z = brute_Z_node(K, d, H, problem_type, chi)

                        results.append(assert_close(name_base + " | old vs new", old_z, new_z))
                        results.append(assert_close(name_base + " | old vs brute", old_z, brute_z, atol=5e-12, rtol=1e-10))

    return results


def run_damped_one_step_tests(rng: np.random.Generator) -> list[CaseResult]:
    results: list[CaseResult] = []

    configs = [
        (2, "assortative"),
        (3, "assortative"),
        (3, "disassortative"),
    ]

    dampings = [0.0, 0.01, 0.2, 1.0]
    chi_kinds = ["random", "almost_uniform", "diagonal_biased", "offdiag_biased", "sparse_soft"]

    for K, problem_type in configs:
        for d in range(2, 9):
            for H in range(1, d + 1):
                for damping in dampings:
                    for kind in chi_kinds:
                        for trial in range(3):
                            chi = random_chi(K, rng, kind)
                            mu = random_mu(K, rng)

                            old_step = old_actual_one_step(K, d, H, problem_type, chi, mu, damping)
                            new_step_fast = new_one_step(K, d, H, problem_type, chi, mu, damping, "fast")
                            new_step_ref = new_one_step(K, d, H, problem_type, chi, mu, damping, "reference")

                            name_base = (
                                f"damped step K={K} {problem_type} d={d} H={H} "
                                f"damping={damping} kind={kind} trial={trial}"
                            )

                            results.append(assert_close(name_base + " | old vs new fast", old_step, new_step_fast))
                            results.append(assert_close(name_base + " | old vs new reference", old_step, new_step_ref))

    return results


def run_factorial_identity_tests() -> list[CaseResult]:
    """
    Tests the combinatorial identity behind old get_factorial_chi.

    For K=3 old assortative/disassortative:
        C(D,r) C(D-r,k) (D-r-k)/D
        =
        (D-1)! / [r! k! (D-1-r-k)!]

    For K=2:
        C(D,r) (D-r)/D
        =
        C(D-1,r)
    """
    results: list[CaseResult] = []

    for d in range(1, 20):
        for H in range(1, d + 1):
            F2 = old2ass.Factorial(d, H)
            for r in range(H - 1, d):
                old_coef = F2.get_factorial_chi(r)
                true_coef = math.comb(d - 1, r)
                results.append(assert_close(f"K=2 factorial identity d={d} H={H} r={r}", old_coef, true_coef))

            F3a = old3ass.Factorial(d, H)
            for r in range(H - 1, d):
                for k in range(0, d - r):
                    old_coef = F3a.get_factorial_chi(r, k)
                    true_coef = math.factorial(d - 1) / (
                        math.factorial(r)
                        * math.factorial(k)
                        * math.factorial(d - 1 - r - k)
                    )
                    results.append(assert_close(
                        f"K=3 ass factorial identity d={d} H={H} r={r} k={k}",
                        old_coef,
                        true_coef,
                    ))

            F3d = old3dis.Factorial(d, H)
            for r in range(0, H):
                if r >= d:
                    continue
                for k in range(0, d - r):
                    old_coef = F3d.get_factorial_chi(r, k)
                    true_coef = math.factorial(d - 1) / (
                        math.factorial(r)
                        * math.factorial(k)
                        * math.factorial(d - 1 - r - k)
                    )
                    results.append(assert_close(
                        f"K=3 dis factorial identity d={d} H={H} r={r} k={k}",
                        old_coef,
                        true_coef,
                    ))

    return results


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------

def summarize(results: list[CaseResult]) -> None:
    failed = [r for r in results if not r.passed]
    passed = len(results) - len(failed)
    max_err = max((r.max_abs_err for r in results), default=0.0)

    print("=" * 80)
    print(f"Total tests: {len(results)}")
    print(f"Passed:      {passed}")
    print(f"Failed:      {len(failed)}")
    print(f"Max abs err: {max_err:.3e}")
    print("=" * 80)

    if failed:
        print("\nFirst 20 failures:\n")
        for r in failed[:20]:
            print("-" * 80)
            print(r.name)
            print(f"max_abs_err = {r.max_abs_err:.3e}")
            print(r.details)

        raise AssertionError(f"{len(failed)} tests failed")
    
def test_mu_solver_independent():
    import numpy as np
    import BP as newbp

    rng = np.random.default_rng(0)

    for K in [2, 3, 4, 5]:
        for d in [3, 5, 7, 10]:
            for trial in range(20):
                chi = rng.random((K, K)) + 0.1
                chi = chi / chi.sum()

                # Generate non-uniform valid targets
                m_target = rng.random(K)
                m_target = m_target / m_target.sum()

                mu = newbp.find_current_mu(
                    K=K,
                    d=d,
                    m_target=m_target,
                    chi=chi,
                    mu0=np.zeros(K),
                    mode="zero",
                    loss="linear",
                )

                m_actual = newbp.compute_m_actual(K, d, mu, chi)

                err = np.max(np.abs(m_actual - m_target))

                print(
                    f"K={K}, d={d}, trial={trial}, "
                    f"target={np.round(m_target, 4)}, "
                    f"actual={np.round(m_actual, 4)}, "
                    f"err={err:.3e}, "
                    f"mu={np.round(mu, 4)}"
                )

                assert err < 1e-7, (
                    f"mu solver failed: K={K}, d={d}, "
                    f"target={m_target}, actual={m_actual}, mu={mu}"
                )

def brute_Z_edge_and_m(K, d, mu, chi):
    W = np.zeros((K, K))

    for a in range(K):
        for b in range(K):
            W[a, b] = (
                np.exp((mu[a] + mu[b]) / d)
                * chi[a, b]
                * chi[b, a]
            )

    Z = W.sum()
    m = W.sum(axis=1) / Z
    return Z, m


def test_Z_edge_and_m_actual():
    import numpy as np
    import BP as newbp

    rng = np.random.default_rng(1)

    for K in [2, 3, 4, 5, 6]:
        for d in [2, 3, 7, 10]:
            for trial in range(100):
                chi = rng.random((K, K)) + 0.1
                chi = chi / chi.sum()

                mu = rng.normal(size=K)
                mu = mu - mu.mean()

                Z_brute, m_brute = brute_Z_edge_and_m(K, d, mu, chi)

                Z_code = newbp.compute_Z_edge(K, d, mu, chi)
                m_code = newbp.compute_m_actual(K, d, mu, chi)

                assert np.allclose(Z_code, Z_brute, atol=1e-12, rtol=1e-10), (
                    Z_code, Z_brute
                )

                assert np.allclose(m_code, m_brute, atol=1e-12, rtol=1e-10), (
                    m_code, m_brute
                )


def run_new_bp_always_zero(K, d, H, problem_type, chi_init, damping, max_iter, threshold):
    cfg = newbp.BPConfig(
        K=K,
        d=d,
        H=H,
        problem_type=problem_type,
        m_target=np.ones(K) / K,
        max_iter=max_iter,
        threshold=threshold,
        damping=damping,
        init_type="manual",
        init_chi=chi_init.copy(),
        mu_mode="always_zero",
        update_path="fast",
        use_wandb=False,
        save_locally=False,
    )
    res = newbp.run_bp(cfg, verbose=0)
    return res


def run_old_bp_always_zero(K, d, H, problem_type, chi_init, damping, max_iter, threshold):
    M = np.ones(K) / K
    mu0 = np.zeros(K)
    settingmu = "always_zero"
    loss = "linear"
    log_every = max_iter + 1
    use_wandb = False

    if K == 2 and problem_type == "assortative":
        F = old2ass.Factorial(d, H)
        chi, mu, iters, total_time, converged = old2ass.run_bp(
            d, H, M, threshold, max_iter, chi_init.copy(), damping,
            mu0, settingmu, log_every, use_wandb, loss, F
        )
        Z_node, Z_edge, phi_RS, m_actual, s = old2ass.compute_quantities(d, H, chi, mu, F)

    elif K == 3 and problem_type == "assortative":
        F = old3ass.Factorial(d, H)
        chi, mu, iters, total_time, converged = old3ass.run_bp(
            d, H, M, threshold, max_iter, chi_init.copy(), damping,
            mu0, settingmu, log_every, use_wandb, loss, F
        )
        Z_node, Z_edge, phi_RS, m_actual, s = old3ass.compute_quantities(d, H, chi, mu, F)

    elif K == 3 and problem_type == "disassortative":
        F = old3dis.Factorial(d, H)
        chi, mu, iters, total_time, converged = old3dis.run_bp(
            d, H, M, threshold, max_iter, chi_init.copy(), damping,
            mu0, settingmu, log_every, use_wandb, loss, F
        )
        Z_node, Z_edge, phi_RS, m_actual, s = old3dis.compute_quantities(d, H, chi, mu, F)

    else:
        raise ValueError((K, problem_type))

    return {
        "chi": chi,
        "mu": mu,
        "iters": iters,
        "converged": converged,
        "Z_node": Z_node,
        "Z_edge": Z_edge,
        "phi_RS": phi_RS,
        "m_actual": m_actual,
        "s": s,
    }


def test_full_old_vs_new_always_zero_runs():
    rng = np.random.default_rng(123)

    configs = [
        (2, "assortative"),
        (3, "assortative"),
        (3, "disassortative"),
    ]

    damping = 0.01
    max_iter = 5000
    threshold = 1e-12

    chi_kinds = [
        "random",
        "almost_uniform",
        "diagonal_biased",
        "offdiag_biased",
        "sparse_soft",
    ]

    for K, problem_type in configs:
        for d in range(3, 9):
            for H in range(1, d + 1):
                for kind in chi_kinds:
                    for trial in range(5):
                        chi_init = random_chi(K, rng, kind)

                        old = run_old_bp_always_zero(
                            K, d, H, problem_type,
                            chi_init, damping, max_iter, threshold
                        )
                        new = run_new_bp_always_zero(
                            K, d, H, problem_type,
                            chi_init, damping, max_iter, threshold
                        )

                        name = (
                            f"K={K}, {problem_type}, d={d}, H={H}, "
                            f"kind={kind}, trial={trial}"
                        )

                        assert np.allclose(new.chi, old["chi"], atol=1e-10, rtol=1e-8), (
                            f"chi mismatch: {name}\n"
                            f"old chi=\n{old['chi']}\n"
                            f"new chi=\n{new.chi}\n"
                            f"diff=\n{new.chi - old['chi']}"
                        )

                        assert np.allclose(new.mu, old["mu"], atol=1e-12, rtol=1e-12), (
                            f"mu mismatch: {name}, old={old['mu']}, new={new.mu}"
                        )

                        assert np.allclose(new.Z_node, old["Z_node"], atol=1e-10, rtol=1e-8), (
                            f"Z_node mismatch: {name}, old={old['Z_node']}, new={new.Z_node}"
                        )

                        assert np.allclose(new.Z_edge, old["Z_edge"], atol=1e-10, rtol=1e-8), (
                            f"Z_edge mismatch: {name}, old={old['Z_edge']}, new={new.Z_edge}"
                        )

                        assert np.allclose(new.phi_RS, old["phi_RS"], atol=1e-10, rtol=1e-8), (
                            f"phi mismatch: {name}, old={old['phi_RS']}, new={new.phi_RS}"
                        )

                        assert np.allclose(new.m_actual, old["m_actual"], atol=1e-10, rtol=1e-8), (
                            f"m_actual mismatch: {name}\n"
                            f"old={old['m_actual']}\n"
                            f"new={new.m_actual}"
                        )

                        assert np.allclose(new.s, old["s"], atol=1e-10, rtol=1e-8), (
                            f"s mismatch: {name}, old={old['s']}, new={new.s}"
                        )

    print("Full old-vs-new always_zero run tests passed.")

def test_K4_update_and_Znode_against_bruteforce():
    rng = np.random.default_rng(777)

    K = 4
    problem_types = ["assortative", "disassortative"]

    for problem_type in problem_types:
        for d in range(2, 8):
            for H in range(1, d + 1):
                for trial in range(20):
                    chi = rng.random((K, K)) + 0.1
                    chi = chi / chi.sum()

                    mu = rng.normal(size=K)
                    mu = mu - mu.mean()

                    raw_fast = newbp._update_chi_fast(K, d, H, problem_type, chi, mu)
                    raw_ref = newbp._update_chi_reference(K, d, H, problem_type, chi, mu)
                    raw_brute = brute_raw_update(K, d, H, problem_type, chi, mu)

                    assert np.allclose(raw_fast, raw_brute, atol=1e-12, rtol=1e-10), (
                        f"K=4 fast update mismatch: {problem_type}, d={d}, H={H}\n"
                        f"fast=\n{raw_fast}\n"
                        f"brute=\n{raw_brute}\n"
                        f"diff=\n{raw_fast - raw_brute}"
                    )

                    assert np.allclose(raw_ref, raw_brute, atol=1e-12, rtol=1e-10), (
                        f"K=4 reference update mismatch: {problem_type}, d={d}, H={H}\n"
                        f"ref=\n{raw_ref}\n"
                        f"brute=\n{raw_brute}\n"
                        f"diff=\n{raw_ref - raw_brute}"
                    )

                    Z_fast = newbp.compute_Z_node(K, d, H, problem_type, chi)
                    Z_brute = brute_Z_node(K, d, H, problem_type, chi)

                    assert np.allclose(Z_fast, Z_brute, atol=1e-12, rtol=1e-10), (
                        f"K=4 Z_node mismatch: {problem_type}, d={d}, H={H}\n"
                        f"new={Z_fast}, brute={Z_brute}"
                    )

    print("K=4 update and Z_node brute-force tests passed.")

def test_full_run_nonuniform_target_after_mu_fix():
    import numpy as np
    import BP as newbp

    cfg = newbp.BPConfig(
        K=3,
        d=7,
        H=4,
        problem_type="assortative",
        m_target=np.array([0.5, 0.3, 0.2]),
        max_iter=50_000,
        threshold=1e-12,
        damping=0.05,
        init_type="almost_unif",
        mu_mode="previous",
        seed=0,
        use_wandb=False,
        save_locally=False,
    )

    res = newbp.run_bp(cfg, verbose=0)

    print("converged:", res.converged)
    print("m_target:", cfg.m_target)
    print("m_actual:", res.m_actual)
    print("mu:", res.mu)
    print("s:", res.s)

    print("chi:")
    print(res.chi)
    print("row sums:", res.chi.sum(axis=1))
    print("col sums:", res.chi.sum(axis=0))
    print("min chi:", res.chi.min())
    print("max chi:", res.chi.max())

    assert np.max(np.abs(res.m_actual - cfg.m_target)) < 1e-6

def compare_old_new_known_negative_entropy_case(
    K,
    d,
    H,
    problem_type,
    init_type="almost_unif",
    seed=0,
    damping=0.01,
    threshold=1e-15,
    max_iter=500_000,
    mu_mode="always_zero",
):
    """
    Compare old and new BP on a known hard/negative-entropy regime.

    This is stricter than formula tests: it compares the full iterative dynamics
    and final observables for a fixed seed and identical initialization.

    Currently supports:
      - K=3 assortative, old3ass
      - K=3 disassortative, old3dis
    """

    assert K == 3

    rng = np.random.default_rng(seed)

    if init_type == "almost_unif":
        # Make initialization explicitly identical for old and new.
        chi_init = np.ones((K, K)) + rng.standard_normal((K, K)) / 100.0
        chi_init = chi_init / chi_init.sum()
    elif init_type == "uniform":
        chi_init = np.ones((K, K), dtype=float)
        chi_init = chi_init / chi_init.sum()
    elif init_type == "gaussian":
        chi_init = rng.random((K, K)) + 1.0
        chi_init = chi_init / chi_init.sum()
    else:
        raise ValueError(f"Unsupported init_type for this test: {init_type}")

    m_target = np.ones(K) / K
    mu0 = np.zeros(K)
    loss = "linear"
    log_every = max_iter + 1
    use_wandb = False

    # -------------------------
    # Old run
    # -------------------------
    if problem_type == "assortative":
        F_old = old3ass.Factorial(d, H)
        old_chi, old_mu, old_iters, old_total_time, old_converged = old3ass.run_bp(
            d,
            H,
            m_target,
            threshold,
            max_iter,
            chi_init.copy(),
            damping,
            mu0,
            mu_mode,  # harmless, only string matters below
            log_every,
            use_wandb,
            loss,
            F_old,
        )
        old_Z_node, old_Z_edge, old_phi, old_m_actual, old_s = old3ass.compute_quantities(
            d, H, old_chi, old_mu, F_old
        )

    elif problem_type == "disassortative":
        F_old = old3dis.Factorial(d, H)
        old_chi, old_mu, old_iters, old_total_time, old_converged = old3dis.run_bp(
            d,
            H,
            m_target,
            threshold,
            max_iter,
            chi_init.copy(),
            damping,
            mu0,
            mu_mode.replace("mu_mode", ""),
            log_every,
            use_wandb,
            loss,
            F_old,
        )
        old_Z_node, old_Z_edge, old_phi, old_m_actual, old_s = old3dis.compute_quantities(
            d, H, old_chi, old_mu, F_old
        )
    else:
        raise ValueError(problem_type)

    # -------------------------
    # New run
    # -------------------------
    cfg = newbp.BPConfig(
        K=K,
        d=d,
        H=H,
        problem_type=problem_type,
        m_target=m_target,
        max_iter=max_iter,
        threshold=threshold,
        damping=damping,
        init_type="manual",
        init_chi=chi_init.copy(),
        mu_mode=mu_mode,
        update_path="fast",
        use_wandb=False,
        save_locally=False,
        seed=seed,
    )

    new_res = newbp.run_bp(cfg, verbose=0)

    # -------------------------
    # Print full diagnostic
    # -------------------------
    print("\n" + "=" * 90)
    print(f"Known negative-entropy case: K={K}, d={d}, H={H}, {problem_type}")
    print(f"seed={seed}, init_type={init_type}, mu_mode={mu_mode}")
    print("-" * 90)

    print("old_converged:", old_converged)
    print("new_converged:", new_res.converged)
    print("old_iters:", old_iters)
    print("new_iters:", new_res.iters)

    print("\nold chi:")
    print(old_chi)
    print("\nnew chi:")
    print(new_res.chi)
    print("\nchi diff:")
    print(new_res.chi - old_chi)
    print("max |chi diff|:", np.max(np.abs(new_res.chi - old_chi)))

    print("\nold mu:", old_mu)
    print("new mu:", new_res.mu)
    print("mu diff:", new_res.mu - old_mu)

    print("\nold Z_node:", old_Z_node)
    print("new Z_node:", new_res.Z_node)
    print("Z_node diff:", new_res.Z_node - old_Z_node)

    print("\nold Z_edge:", old_Z_edge)
    print("new Z_edge:", new_res.Z_edge)
    print("Z_edge diff:", new_res.Z_edge - old_Z_edge)

    print("\nold phi_RS:", old_phi)
    print("new phi_RS:", new_res.phi_RS)
    print("phi diff:", new_res.phi_RS - old_phi)

    print("\nold m_actual:", old_m_actual)
    print("new m_actual:", new_res.m_actual)
    print("m diff:", new_res.m_actual - old_m_actual)

    print("\nold entropy s:", old_s)
    print("new entropy s:", new_res.s)
    print("s diff:", new_res.s - old_s)
    print("=" * 90)

    # -------------------------
    # Assertions
    # -------------------------
    assert np.allclose(new_res.chi, old_chi, atol=1e-9, rtol=1e-7), (
        f"chi mismatch in K={K}, d={d}, H={H}, {problem_type}"
    )

    assert np.allclose(new_res.Z_node, old_Z_node, atol=1e-10, rtol=1e-8), (
        f"Z_node mismatch in K={K}, d={d}, H={H}, {problem_type}"
    )

    assert np.allclose(new_res.Z_edge, old_Z_edge, atol=1e-10, rtol=1e-8), (
        f"Z_edge mismatch in K={K}, d={d}, H={H}, {problem_type}"
    )

    assert np.allclose(new_res.phi_RS, old_phi, atol=1e-9, rtol=1e-7), (
        f"phi_RS mismatch in K={K}, d={d}, H={H}, {problem_type}"
    )

    assert np.allclose(new_res.s, old_s, atol=1e-9, rtol=1e-7), (
        f"entropy mismatch in K={K}, d={d}, H={H}, {problem_type}"
    )

    assert new_res.s < 0, (
        f"Expected negative entropy, got s={new_res.s} "
        f"for K={K}, d={d}, H={H}, {problem_type}"
    )

def test_known_negative_entropy_old_vs_new():
    cases = [
        {
            "K": 3,
            "d": 8,
            "H": 7,
            "problem_type": "assortative",
        },
        {
            "K": 3,
            "d": 8,
            "H": 1,
            "problem_type": "disassortative",
        },
    ]

    for case in cases:
        for seed in range(5):
            compare_old_new_known_negative_entropy_case(
                **case,
                seed=seed,
                init_type="almost_unif",
                damping=0.01,
                threshold=1e-15,
                max_iter=500_000,
                mu_mode="always_zero",
            )

def test_known_negative_entropy_uniform_closed_form():
    K = 3
    d = 8
    chi = np.ones((K, K), dtype=float)
    chi = chi / chi.sum()
    mu = np.zeros(K)

    alpha = 1.0 / 9.0
    beta = 2.0 / 9.0

    # -------------------------
    # Assortative: H=7
    # -------------------------
    H = 7
    problem_type = "assortative"

    Z_code = newbp.compute_Z_node(K, d, H, problem_type, chi)

    Z_expected = 3.0 * sum(
        math.comb(d, r) * (alpha ** r) * (beta ** (d - r))
        for r in range(H, d + 1)
    )

    assert np.allclose(Z_code, Z_expected, atol=1e-15, rtol=1e-12), (
        f"assortative uniform Z_node mismatch: code={Z_code}, expected={Z_expected}"
    )

    Ze = newbp.compute_Z_edge(K, d, mu, chi)
    phi = newbp.compute_phi_RS(d, Z_code, Ze)
    s = newbp.compute_entropy(phi, mu, np.ones(K) / K)

    print("\nUniform closed form: K=3, d=8, H=7 assortative")
    print("Z_node:", Z_code)
    print("Z_edge:", Ze)
    print("phi=s:", s)

    assert s < 0

    # -------------------------
    # Disassortative: H=1
    # -------------------------
    H = 1
    problem_type = "disassortative"

    Z_code = newbp.compute_Z_node(K, d, H, problem_type, chi)

    Z_expected = 3.0 * sum(
        math.comb(d, r) * (alpha ** r) * (beta ** (d - r))
        for r in range(0, H)
    )

    assert np.allclose(Z_code, Z_expected, atol=1e-15, rtol=1e-12), (
        f"disassortative uniform Z_node mismatch: code={Z_code}, expected={Z_expected}"
    )

    Ze = newbp.compute_Z_edge(K, d, mu, chi)
    phi = newbp.compute_phi_RS(d, Z_code, Ze)
    s = newbp.compute_entropy(phi, mu, np.ones(K) / K)

    print("\nUniform closed form: K=3, d=8, H=1 disassortative")
    print("Z_node:", Z_code)
    print("Z_edge:", Ze)
    print("phi=s:", s)

    assert s < 0

def diagnostic_K3_D7_H4_entropy_zero():
    import numpy as np
    import BP as newbp

    configs = [
        ("uniform, mu=0", "uniform", "always_zero"),
        ("almost_unif, mu=0", "almost_unif", "always_zero"),
        ("almost_unif, solved mu", "almost_unif", "previous"),
    ]

    for label, init_type, mu_mode in configs:
        print("\n" + "=" * 80)
        print(label)

        for seed in range(5):
            cfg = newbp.BPConfig(
                K=3,
                d=7,
                H=4,
                problem_type="assortative",
                m_target=np.array([1/3, 1/3, 1/3]),
                max_iter=500_000,
                threshold=1e-14,
                damping=0.01,
                init_type=init_type,
                mu_mode=mu_mode,
                update_path="fast",
                seed=seed,
                use_wandb=False,
                save_locally=False,
            )

            res = newbp.run_bp(cfg, verbose=0)

            print(
                f"seed={seed}, converged={res.converged}, "
                f"iters={res.iters}, "
                f"m={np.round(res.m_actual, 6)}, "
                f"s={res.s:.12g}, "
                f"phi={res.phi_RS:.12g}, "
                f"min_chi={res.chi.min():.2e}, "
                f"max_chi={res.chi.max():.2e}"
            )

def main() -> None:
    rng = np.random.default_rng(12345)
    all_results: list[CaseResult] = []

    groups = [
        # ("factorial coefficient identities", run_factorial_identity_tests),
        # ("raw update comparisons", lambda: run_update_tests(rng)),
        # ("Z_node comparisons", lambda: run_Z_node_tests(rng)),
        # ("damped one-step comparisons", lambda: run_damped_one_step_tests(rng)),
        # ("Z_edge and m_actual brute-force comparisons", test_Z_edge_and_m_actual),
        # ("full old-vs-new always_zero runs", test_full_old_vs_new_always_zero_runs),
        # ("mu solver independent consistency", test_mu_solver_independent),
        # ("K=4 update and Z_node brute-force", test_K4_update_and_Znode_against_bruteforce),
        # ("known negative entropy old-vs-new cases", test_known_negative_entropy_old_vs_new),
        # ("known negative entropy uniform closed-form", test_known_negative_entropy_uniform_closed_form),
        # ("known negative entropy old-vs-new cases", test_known_negative_entropy_old_vs_new),
        ("diagnostic_K3_D7_H4_entropy_zero", diagnostic_K3_D7_H4_entropy_zero),
    ]

    for group_name, fn in groups:
        print(f"\nRunning: {group_name}")
        try:
            result = fn()

            # Some older test groups return list[CaseResult].
            # The new assert-based tests return None if they pass.
            if result is None:
                print("=" * 80)
                print("Passed")
                print("=" * 80)
            else:
                summarize(result)
                all_results.extend(result)

        except Exception:
            print(f"\nERROR while running group: {group_name}")
            traceback.print_exc()
            raise

    print("\nALL TEST GROUPS PASSED.")


if __name__ == "__main__":
    main()