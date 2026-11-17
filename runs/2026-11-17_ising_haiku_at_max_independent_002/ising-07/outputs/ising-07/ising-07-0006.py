import sys, glob
import numpy as np
sys.path.insert(0, '/work')
import wolff_prod as wp

files = sorted(glob.glob('/work/prod/blocks_*.npz'))
M2=[];M4=[];CNT=[]
for f in files:
    d = np.load(f); M2.append(d['m2']); M4.append(d['m4']); CNT.append(d['cnt'])
M2 = np.concatenate(M2); M4 = np.concatenate(M4); CNT = np.concatenate(CNT)
ntot = CNT.sum()
nb = len(CNT)
print("blocks:", nb, "steps per chain:", ntot)
S2 = M2.sum(0)/ntot; S4 = M4.sum(0)/ntot          # (R,)
s2 = S2.reshape(wp.NT, wp.NC); s4 = S4.reshape(wp.NT, wp.NC)

rng = np.random.default_rng(3000)
print(" T      U(full)      Uerr_jack   Uerr_subset(8x32)  Uerr_timeblk(20)   U_err_max_ratio")
for g in range(wp.NT):
    a2 = s2[g]; a4 = s4[g]
    U, Ue = wp.jackknife(a2, a4)
    # (a) random subsets of 32 chains, 8 disjoint groups, averaged over 4 random partitions
    ests = []
    for rep in range(4):
        perm = rng.permutation(wp.NC)
        Ug = []
        for k in range(8):
            idx = perm[k*32:(k+1)*32]
            Ug.append(1 - (a4[idx].mean())/(3*(a2[idx].mean())**2))
        ests.append(np.std(Ug, ddof=1)/np.sqrt(8))
    Ue_sub = np.mean(ests)
    # (b) time blocks: 20 contiguous groups of block-sums
    bidx = np.array_split(np.arange(nb), 20)
    Ub = []
    for ix in bidx:
        n_ = CNT[ix].sum()
        b2 = M2[ix].sum(0)/n_; b4 = M4[ix].sum(0)/n_
        b2 = b2.reshape(wp.NT, wp.NC)[g]; b4 = b4.reshape(wp.NT, wp.NC)[g]
        Ub.append(1 - b4.mean()/(3*b2.mean()**2))
    Ub = np.array(Ub)
    Ue_t = Ub.std(ddof=1)/np.sqrt(len(Ub))
    print(f"{wp.Ts[g]:.2f} {U:.7f}  {Ue:.6f}     {Ue_sub:.6f}          {Ue_t:.6f}         {Ue_t/Ue:.2f}")
