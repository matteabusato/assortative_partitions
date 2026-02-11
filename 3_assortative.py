import numpy as np
import math
np.random.seed(42)

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

def find_current_mu(D, M, chi):
    # find mu such that m_actual and M are as close as possible
    Z_edge = compute_Z_edge(D, mu)
    m_actual = compute_m_actual(D, mu, Z_edge, chi)
    # mu = argmin |m(\mu, \chi)-M|
    
def update_chi(D, H, M, THRESHOLD, MAX_ITER, chi, damping):
    mu = find_current_mu(D, M, chi)

    for i in range(3):
        for j in range(3):
            second_term = 0
            f1, f2, f3 = assign_f(i)
            for r in range(H-(i==j), D-1):
                for k in range(D-i-r):
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

def compute_Z_edge(D, mu):
    Z_edge = 0
    for i in range(3):
        for j in range(3):
            Z_edge += np.exp((1/D)*(mu[i]*mu[j])) * chi[i, j] * chi[j, i]
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
    Z_edge = compute_Z_edge(D, mu)
    phi_RS = compute_phi_RS(D, Z_node, Z_edge)
    m_actual = compute_m_actual(D, mu, Z_edge, chi)
    s = compute_entropy(phi_RS, mu, m_actual)
    return Z_node, Z_edge, phi_RS, m_actual, s

def run_bp(D, H, M, THRESHOLD, MAX_ITER, chi, damping):
    iter = 0
    while iter < MAX_ITER:
        chi_new, mu = update_chi(D, H, M, THRESHOLD, MAX_ITER, chi, damping)
        
        diff = np.max(np.abs(chi_new - chi))
        chi = chi_new

        if diff < THRESHOLD:
            break
        iter += 1

    return chi, mu, iter


if __name__ == "__main__":
    # K = 3  ->  3 groups in the partition
    N = 15
    D = 7
    H = 2
    M = np.array([1/3, 1/3, 1/3])
    THRESHOLD = 1e-10
    MAX_ITER = 100000
    chi = np.zeros((3, 3)) # (K, K)
    damping = 0.1

    chi, mu, time = run_bp(D, H, M, THRESHOLD, MAX_ITER, chi, damping)

    Z_node, Z_edge, phi_RS, m_actual, s = compute_quantities(D, H, chi, mu)

    print("Found fixed point chi = \n", chi, " after time: ", time)
    print("Z_node = ", Z_node, ", Z_edge = ", Z_edge, ", phi_RS = ", phi_RS, ", actual magentization = ", m_actual, ", entropy = ", s)
    print("The number of solution with magentization close to target ", M, " is ", np.exp(s*N))
