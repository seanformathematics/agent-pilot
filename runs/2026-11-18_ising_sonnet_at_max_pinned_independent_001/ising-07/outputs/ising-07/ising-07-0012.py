
code = r'''
import numpy as np

class MetroBatch:
    """R independent N x N periodic Ising lattices, checkerboard Metropolis (J=1,k_B=1)."""
    def __init__(self, N, R, T, seed, start):
        self.N, self.R, self.T = N, R, float(T)
        self.rng = np.random.default_rng(seed)
        s = self.rng.integers(0, 2, size=(R, N, N), dtype=np.int8) * 2 - 1
        flags = np.asarray(start, dtype=bool)
        s[flags] = 1
        self.s = s
        ii, jj = np.indices((N, N))
        par = ((ii + jj) & 1).astype(bool)
        self.par = [~par, par]
        # s*nn in {-4,-2,0,2,4}; dE = 2*s*nn ; acceptance min(1, exp(-dE/T))
        self.acc = np.minimum(1.0, np.exp(-2.0 * np.array([-4, -2, 0, 2, 4]) / self.T)).astype(np.float32)

    def sweep(self):
        s, rng, R, N = self.s, self.rng, self.R, self.N
        for par in self.par:
            nn = np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2)
            k = (s * nn + 4) >> 1
            a = self.acc[k]
            r = rng.random((R, N, N), dtype=np.float32)
            flip = (r < a) & par
            s *= (1 - 2 * flip.astype(np.int8))
        return s.reshape(R, -1).sum(axis=1, dtype=np.int32)
'''
open('/work/metro.py','w').write(code)
import sys, time
sys.path.insert(0,'/work')
import numpy as np
from metro import MetroBatch
# quick correctness: N=4 vs exact
N=4; Ns=16
states = np.arange(2**Ns, dtype=np.uint32)
bits = ((states[:,None] >> np.arange(Ns, dtype=np.uint32)[None,:]) & 1).astype(np.int8)*2-1
S = bits.reshape(-1,N,N)
E = -(S*np.roll(S,-1,axis=2)).sum(axis=(1,2)) - (S*np.roll(S,-1,axis=1)).sum(axis=(1,2))
m = S.sum(axis=(1,2)).astype(np.float64)/Ns
T=2.5
w=np.exp(-(E-E.min())/T); w/=w.sum()
m2e=(w*m**2).sum(); m4e=(w*m**4).sum(); Ue=1-m4e/(3*m2e**2); Ee=(w*E).sum()/Ns
sim=MetroBatch(4,500,T,seed=11,start=np.arange(500)%2)
for _ in range(200): sim.sweep()
Ms=np.array([sim.sweep() for _ in range(4000)],float)/Ns   # (4000,500)
c2=(Ms**2).mean(axis=0); c4=(Ms**4).mean(axis=0); n=500
U=1-c4.mean()/(3*c2.mean()**2)
loo2=(c2.sum()-c2)/(n-1); loo4=(c4.sum()-c4)/(n-1); Ul=1-loo4/(3*loo2**2)
err=np.sqrt((n-1)/n*((Ul-Ul.mean())**2).sum())
print("Metropolis N=4 T=2.5: U=%.5f+-%.5f exact %.5f (dev %.2f sigma)"%(U,err,Ue,(U-Ue)/err))

# speed at N=16
for R in (500,):
    sim=MetroBatch(16,R,2.3,seed=5,start=np.arange(R)%2)
    for _ in range(10): sim.sweep()
    t=time.time(); n=40
    for _ in range(n): sim.sweep()
    dt=(time.time()-t)/n
    print("N=16 R=%d: %.2f ms/sweep -> %.0f sweeps-replicas/s"%(R,dt*1e3,R/dt))
