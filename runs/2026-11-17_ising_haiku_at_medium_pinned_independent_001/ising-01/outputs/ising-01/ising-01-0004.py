
import sys, time
sys.path.insert(0, '/work')
import numpy as np
from sw import *

N = 4
NN = N*N
# enumerate all configs
idx = np.arange(2**NN)
bits = ((idx[:, None] >> np.arange(NN)) & 1).astype(np.int8)
confs = (2*bits - 1).reshape(-1, N, N).astype(np.int8)
Ecf = -((confs * np.roll(confs, -1, axis=2)).reshape(-1, NN).sum(1) + (confs * np.roll(confs, -1, axis=1)).reshape(-1, NN).sum(1))
m = confs.reshape(-1, NN).mean(1)
for T in [1.0, 2.0, 2.5, 3.5]:
    w = np.exp(-(Ecf - Ecf.min()) / T)
    Z = w.sum()
    eE = (w * Ecf).sum() / Z / NN
    m2 = (w * m**2).sum() / Z
    m4 = (w * m**4).sum() / Z
    Uex = 1 - m4 / (3 * m2**2)
    # SW estimate
    rng = np.random.default_rng(7)
    R = 2000
    S = make_state(R, N, rng)
    for _ in range(50):
        S = sw_step(S, T, rng, N)
    acc = np.zeros(4); cnt = 0
    for k in range(400):
        S = sw_step(S, T, rng, N)
        mm, ee = measure(S, N)
        acc += [ee.mean(), (mm**2).mean(), (mm**4).mean(), 0]
        cnt += 1
    acc /= cnt
    Usw = 1 - acc[2] / (3 * acc[1]**2)
    print(f"T={T}: exact E/N2={eE:.5f} SW={acc[0]:.5f} | exact m2={m2:.5f} SW={acc[1]:.5f} | exact m4={m4:.5f} SW={acc[2]:.5f} | U exact={Uex:.5f} SW={Usw:.5f}")
