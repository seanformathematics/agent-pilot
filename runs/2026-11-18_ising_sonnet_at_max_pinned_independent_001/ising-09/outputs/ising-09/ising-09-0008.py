
import sys, time
sys.path.insert(0, "/work")
code = r'''
import numpy as np
from run import jackknife_U

class Metro:
    """Checkerboard single-spin-flip Metropolis, R independent replicas of an NxN torus (N even)."""
    def __init__(self, N, R, seed, start="hot"):
        self.N=N; self.R=R
        self.rng = np.random.default_rng(seed)
        if start=="hot":
            self.s = (self.rng.integers(0,2,size=(R,N,N),dtype=np.int8)*2-1).astype(np.int8)
        else:
            self.s = np.ones((R,N,N), dtype=np.int8)
        x,y = np.indices((N,N))
        self.mask = [((x+y)%2==c)[None,:,:] for c in (0,1)]
    def sweep(self, T):
        s = self.s; rng = self.rng
        lut = np.ones(17, dtype=np.float32)
        lut[12] = np.exp(-4.0/T); lut[16] = np.exp(-8.0/T)     # dE=+4 -> idx 12 ; dE=+8 -> idx 16
        for c in (0,1):
            nn = np.roll(s,1,axis=1)+np.roll(s,-1,axis=1)+np.roll(s,1,axis=2)+np.roll(s,-1,axis=2)
            dE = 2*s*nn
            u = rng.random(s.shape, dtype=np.float32)
            flip = (u < lut[dE+8]) & self.mask[c]
            s = np.where(flip, -s, s).astype(np.int8)
        self.s = s

def run_metro(N, T, R, E, S, L, seed, start):
    sim = Metro(N, R, seed, start)
    for _ in range(E): sim.sweep(T)
    nb = S//L
    acc = np.zeros((2,R,nb))
    for b in range(nb):
        for k in range(L):
            sim.sweep(T)
            m = sim.s.reshape(R,-1).sum(axis=1, dtype=np.int64)/(N*N)
            m2 = m*m
            acc[0,:,b] += m2; acc[1,:,b] += m2*m2
    return acc/L
'''
open("/work/metro.py","w").write(code)
import numpy as np
from metro import *
# timing test
t=time.time()
sim = Metro(16, 1000, 5, "hot")
for _ in range(20): sim.sweep(2.3)
dt=(time.time()-t)/20
print(f"Metropolis: {dt*1e3:.2f} ms/sweep for R=1000 -> {1000/dt:.0f} replica-sweeps/s")
