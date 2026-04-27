import numpy as np
from assortative_partitions.deprecated.pd_3_assortative import population_dynamics_3ass

my_chi = np.array([
    [
      0.16432256810879467,
      0.08450537753499858,
      0.08450537785183071
    ],
    [
      0.08450538717654685,
      0.1643225755268645,
      0.0845053860714727
    ],
    [
      0.08450538190925244,
      0.08450538048734998,
      0.16432256533288964
    ]
  ])

my_chi = my_chi / my_chi.sum()

pd = population_dynamics_3ass(
    d=7, H=4,
    m_parisi=1.0,
    mu=np.zeros(3),
    M=10000,
    init_type='bp_fixed_point_noisy',
    init_chi_bp=my_chi,
    init_noise=1e-8,
    max_iter=2000, tol=0.05,
    sampling_threshold=1000, sampling_interval=50,
    seed=0,
)
pd.run(verbose=2)
print(pd)
pd.save()