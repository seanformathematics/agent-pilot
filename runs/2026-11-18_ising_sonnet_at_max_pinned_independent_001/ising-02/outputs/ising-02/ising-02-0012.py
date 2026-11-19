
import sys, time
sys.path.insert(0,"/work")
import numpy as np
from ising import wolff_run
from metro import metropolis_batch
from analysis import *

N=4; n=N*N
# exact enumeration
confs = ((np.arange(2**n)[:,None] >> np.arange(n)[None,:]) & 1).astype(np.int8)*2-1   # (65536,16)
S = confs.reshape(-1,N,N).astype(np.int32)
E = -(S*np.roll(S,1,1)).sum((1,2)) - (S*np.roll(S,1,2)).sum((1,2))     # each bond once
Msum = S.sum((1,2))
def exact(T):
    w = np.exp(-(E-E.min())/T); w/=w.sum()
    m = Msum/n
    m2 = (w*m**2).sum(); m4=(w*m**4).sum()
    return m2, m4, 1-m4/(3*m2**2)

for T in [1.5, 2.3, 2.8]:
    m2e, m4e, Ue = exact(T)
    # Wolff: 10 replicas x 300k flips, jackknife over replicas x blocks
    ms2=[]; ms4=[]
    for r in range(10):
        M, cb, _ = wolff_run(N, T, 777000+r, 2000, 300000, "hot" if r%2 else "cold")
        a,b = moments(M, N); ms2.append(a); ms4.append(b)
    ms2=np.array(ms2); ms4=np.array(ms4)
    B=20; L=ms2.shape[1]//B
    S2b = ms2.reshape(10,B,L).sum(2).reshape(-1); S4b = ms4.reshape(10,B,L).sum(2).reshape(-1)
    Uw, ew, _ = jackknife_U(S2b, S4b, np.full(10*B, L))
    # Metropolis
    S2m, S4m, cnt = metropolis_batch(N, T, 256, 1000, 20000, seed=31, init="hot")
    Um, em, _ = jackknife_U(S2m, S4m, np.full(256, cnt))
    print(f"N=4 T={T}: exact U={Ue:.6f} (m2={m2e:.5f}, m4={m4e:.5f}) | Wolff U={Uw:.6f}+-{ew:.6f} z={(Uw-Ue)/ew:+.2f} | Metropolis U={Um:.6f}+-{em:.6f} z={(Um-Ue)/em:+.2f}", flush=True)
