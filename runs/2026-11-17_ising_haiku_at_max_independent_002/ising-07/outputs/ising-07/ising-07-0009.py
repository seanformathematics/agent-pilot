import sys, glob
import numpy as np
sys.path.insert(0, '/work')
import wolff_prod as wp
files = sorted(glob.glob('/work/prod/blocks_*.npz'))
M2=[];M4=[];CNT=[]
for f in files:
    d = np.load(f); M2.append(d['m2']); M4.append(d['m4']); CNT.append(d['cnt'])
M2=np.concatenate(M2); M4=np.concatenate(M4); CNT=np.concatenate(CNT).astype(float)
nb = len(CNT); ntot = CNT.sum()
rng = np.random.default_rng(4000)
print(" T      U        jackknife  boot_chains  boot_timeblocks(401 blk)")
for g in range(wp.NT):
    sl = slice(g*wp.NC, (g+1)*wp.NC)
    a2 = M2[:, sl].sum(0)/ntot; a4 = M4[:, sl].sum(0)/ntot   # per-chain means (NC,)
    U, Ue = wp.jackknife(a2, a4)
    # bootstrap over chains
    idx = rng.integers(0, wp.NC, size=(2000, wp.NC))
    Ub_c = 1 - a4[idx].mean(1)/(3*a2[idx].mean(1)**2)
    se_c = Ub_c.std(ddof=1)
    # block bootstrap over time: multinomial weights over the nb blocks, same weights for all chains
    w = rng.multinomial(nb, np.ones(nb)/nb, size=1000).astype(float)   # (1000, nb)
    Wn = w @ CNT                                                       # total steps in resample
    W2 = w @ M2[:, sl]; W4 = w @ M4[:, sl]                             # (1000, NC)
    Ub_t = 1 - (W4/Wn[:,None]).mean(1)/(3*((W2/Wn[:,None]).mean(1))**2)
    se_t = Ub_t.std(ddof=1)
    print(f"{wp.Ts[g]:.2f} {U:.7f} {Ue:.3e}   {se_c:.3e}    {se_t:.3e}")
