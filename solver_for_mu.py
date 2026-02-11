import numpy as np
from scipy.optimize import least_squares

D = 7
chi = np.array([[0.3, 0.3, 0.4], [0.3, 0.3, 0.4], [0.4, 0.4, 0.2]])
m_star = np.array([1, 1/3, 1/3, 1/3])

def m_values(mu):
    mu1, mu2, mu3 = mu
    return np.array([
        -mu1 * (np.exp((2/D)*mu1)*(chi[0, 0]**2)+ np.exp((1/D)*(mu1+mu2))*(chi[0, 1]*chi[1, 0]) + np.exp((1/D)*(mu1+mu3))*(chi[0, 2]*chi[2, 0])) 
        / (np.exp((2/D)*mu1)*(chi[0,0]**2) + np.exp((2/D)*mu2)*(chi[1,1]**2) + np.exp((2/D)*mu3)*(chi[2,2]**2) + 2*(np.exp((1/D)*(mu1+mu2))*chi[0,1]*chi[1,0]) + 2*(np.exp((1/D)*(mu2+mu3))*chi[1,2]*chi[2,1]) + 2*(np.exp((1/D)*(mu3+mu1)) *chi[2,0]*chi[0,2]) ) 
        + -mu2 * (np.exp((2/D)*mu2)*(chi[1, 1]**2)+ np.exp((1/D)*(mu2+mu1))*(chi[1, 0]*chi[0, 1]) + np.exp((1/D)*(mu2+mu3))*(chi[1, 2]*chi[2, 1])) 
        / (np.exp((2/D)*mu1)*(chi[0,0]**2) + np.exp((2/D)*mu2)*(chi[1,1]**2) + np.exp((2/D)*mu3)*(chi[2,2]**2) + 2*(np.exp((1/D)*(mu1+mu2))*chi[0,1]*chi[1,0]) + 2*(np.exp((1/D)*(mu2+mu3))*chi[1,2]*chi[2,1]) + 2*(np.exp((1/D)*(mu3+mu1)) *chi[2,0]*chi[0,2]) )
        + -mu3 * (np.exp((2/D)*mu3)*(chi[2, 2]**2)+ np.exp((1/D)*(mu3+mu1))*(chi[2, 0]*chi[0, 2]) + np.exp((1/D)*(mu3+mu2))*(chi[2, 1]*chi[1, 2])) 
        / (np.exp((2/D)*mu1)*(chi[0,0]**2) + np.exp((2/D)*mu2)*(chi[1,1]**2) + np.exp((2/D)*mu3)*(chi[2,2]**2) + 2*(np.exp((1/D)*(mu1+mu2))*chi[0,1]*chi[1,0]) + 2*(np.exp((1/D)*(mu2+mu3))*chi[1,2]*chi[2,1]) + 2*(np.exp((1/D)*(mu3+mu1)) *chi[2,0]*chi[0,2]) ),
        -mu1 * (np.exp((2/D)*mu1)*(chi[0, 0]**2)+ np.exp((1/D)*(mu1+mu2))*(chi[0, 1]*chi[1, 0]) + np.exp((1/D)*(mu1+mu3))*(chi[0, 2]*chi[2, 0])) 
        / (np.exp((2/D)*mu1)*(chi[0,0]**2) + np.exp((2/D)*mu2)*(chi[1,1]**2) + np.exp((2/D)*mu3)*(chi[2,2]**2) + 2*(np.exp((1/D)*(mu1+mu2))*chi[0,1]*chi[1,0]) + 2*(np.exp((1/D)*(mu2+mu3))*chi[1,2]*chi[2,1]) + 2*(np.exp((1/D)*(mu3+mu1)) *chi[2,0]*chi[0,2]) ),
        -mu2 * (np.exp((2/D)*mu2)*(chi[1, 1]**2)+ np.exp((1/D)*(mu2+mu1))*(chi[1, 0]*chi[0, 1]) + np.exp((1/D)*(mu2+mu3))*(chi[1, 2]*chi[2, 1])) 
        / (np.exp((2/D)*mu1)*(chi[0,0]**2) + np.exp((2/D)*mu2)*(chi[1,1]**2) + np.exp((2/D)*mu3)*(chi[2,2]**2) + 2*(np.exp((1/D)*(mu1+mu2))*chi[0,1]*chi[1,0]) + 2*(np.exp((1/D)*(mu2+mu3))*chi[1,2]*chi[2,1]) + 2*(np.exp((1/D)*(mu3+mu1)) *chi[2,0]*chi[0,2]) ),
        -mu3 * (np.exp((2/D)*mu3)*(chi[2, 2]**2)+ np.exp((1/D)*(mu3+mu1))*(chi[2, 0]*chi[0, 2]) + np.exp((1/D)*(mu3+mu2))*(chi[2, 1]*chi[1, 2])) 
        / (np.exp((2/D)*mu1)*(chi[0,0]**2) + np.exp((2/D)*mu2)*(chi[1,1]**2) + np.exp((2/D)*mu3)*(chi[2,2]**2) + 2*(np.exp((1/D)*(mu1+mu2))*chi[0,1]*chi[1,0]) + 2*(np.exp((1/D)*(mu2+mu3))*chi[1,2]*chi[2,1]) + 2*(np.exp((1/D)*(mu3+mu1)) *chi[2,0]*chi[0,2]) ),
    ], dtype=float)

def residuals(mu):
    return m_values(mu) - m_star 

mu0 = np.zeros(3)
res = least_squares(residuals, mu0, method="trf")

print(res.success, res.message)
print("mu* =", res.x)
print("m(mu*) =", m_values(res.x))
print("target =", m_star)
print("residuals =", residuals(res.x))
print("sumsq =", np.sum(res.fun**2))