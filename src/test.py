import numpy as np
from BP import _update_chi_reference, _update_chi_fast

def test_fast_equals_reference():
    rng = np.random.default_rng(0)

    for K in [2, 3, 4, 5]:
        for d in [3, 5, 7]:
            for H in range(0, d + 1):
                for problem_type in ["assortative", "disassortative"]:
                    chi = rng.random((K, K)) + 0.1
                    chi = chi / chi.sum()
                    mu = rng.normal(size=K)
                    mu = mu - mu.mean()

                    ref = _update_chi_reference(K, d, H, problem_type, chi, mu)
                    fast = _update_chi_fast(K, d, H, problem_type, chi, mu)

                    err = np.max(np.abs(ref - fast))
                    if err > 1e-10:
                        print("Mismatch:", K, d, H, problem_type, err)
                        return False
    return True

print(test_fast_equals_reference())