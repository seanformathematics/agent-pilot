
import sys, time
sys.path.insert(0, '/work')
import numpy as np
from sw import *
N = 4; NN = 16
idx = np.arange(2**NN)
bits = ((idx[:, None] >> np.arange(NN)) & 1).astype(np.int8)
confs = (2*bits - 1).reshape(-1, N, N).astype(np.int8)
Ecf = -((confs * np.roll(confs, -1, axis=2)).reshape(-1, NN).sum(1) + (confs * np.roll(confs, -1, axis=1)).reshape(-1, NN).sum(1))
m = confs.reshape(-1, NN).mean(1)
def jack(x):
    R=len(x); return x.mean(), np.sqrt((R-1)/R*np.sum((x-x.mean())**2))
for T in [2.5, 3.5]:
    w = np.exp(-(Ecf - Ecf.min()) / T); Z = w.sum()
    eE = (w * Ecf).sum() / Z / NN
    m2 = (w * m**2).sum() / Z
    rng = np.random.default_rng(99)
    R = 20000
    S = make_state(R, N, rng)
    for _ in range(30): S = sw_step(S, T, rng, N)
    E_acc = np.zeros(R); M_acc = np.zeros(R); c=0
    t0=time.time()
    for k in range(100):
        S = sw_step(S, T, rng, N)
        mm, ee = measure(S, N)
        E_acc += ee; M_acc += mm**2; c+=1
    E_acc/=c; M_acc/=c
    em, ee_ = jack(E_acc); mm_, mme = jack(M_acc)
    print(f"4x4 T={T}: exact E={eE:.5f}  SW E={em:.5f} +- {ee_:.5f}; exact m2={m2:.5f} SW={mm_:.5f} +- {mme:.5f}  ({time.time()-t0:.1f}s)")
