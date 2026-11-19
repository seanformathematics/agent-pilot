
import sys, time
import numpy as np
sys.path.insert(0,'/work')
from ana import jackknife_U

metro = r'''
import numpy as np
class MetroReplicas:
    def __init__(self, N, R, T, rng):
        assert N % 2 == 0
        self.N, self.R, self.T, self.rng = N, R, T, rng
        self.s = (2 * rng.integers(0, 2, size=(R, N, N), dtype=np.int8) - 1).astype(np.int8)
        ii, jj = np.indices((N, N))
        self.mask = [((ii + jj) % 2 == 0)[None], ((ii + jj) % 2 == 1)[None]]
        self.table = np.minimum(1.0, np.exp(-np.arange(-8, 9) / T))   # acceptance prob for dE = -8..8
    def sweep(self):
        s, rng = self.s, self.rng
        for par in (0, 1):
            nn = np.roll(s, 1, axis=1) + np.roll(s, -1, axis=1) + np.roll(s, 1, axis=2) + np.roll(s, -1, axis=2)
            dE = 2 * s * nn
            acc = (rng.random(s.shape) < self.table[dE + 8]) & self.mask[par]
            s[acc] *= -1
'''
open('/work/metro.py','w').write(metro)
from metro import MetroReplicas

N=16; nsite=N*N
R=1000; nburn=500; nmeas=4000
for iT,T in enumerate([2.3,2.5,2.8]):
    rng=np.random.default_rng(5150+iT)
    sim=MetroReplicas(N,R,T,rng)
    t0=time.time()
    for _ in range(nburn): sim.sweep()
    S2=np.zeros(R); S4=np.zeros(R)
    # also store block sums for autocorrelation insight
    for _ in range(nmeas):
        sim.sweep()
        mm=sim.s.sum(axis=(1,2),dtype=np.int64)/nsite
        S2+=mm**2; S4+=mm**4
    U,err,Ubc=jackknife_U(S2,S4,nmeas)
    print(f"Metropolis N=16 T={T}: U={U:.4f}+-{err:.4f}  <m2>={S2.sum()/(R*nmeas):.4f}  [{time.time()-t0:.0f}s]")
print("SW pilot for comparison: T=2.3 U=0.5924+-0.0004 ; T=2.5 U=0.3848+-0.0010 ; T=2.8 U=0.1462+-0.0014")
