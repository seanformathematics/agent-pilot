
import sys
sys.path.insert(0,'/work')
import numpy as np
from prod import *
Ts = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]

print("Burn-in: <m^2>_impr (hot / cold), averaged over 500 chains each")
tt=[0,1,2,3,5,8,12,20,40,70,99]
print("T    "+" ".join("t=%-3d      "%t for t in tt))
for T in (1.5,1.8,2.0,2.3,2.5,2.8):
    z=np.load("/work/prod_T%.1f.npz"%T)
    b=z['burn'].mean(axis=0)   # (100,2,2): [t, hot/cold, d/i]
    print("%.1f  "%T+" ".join("%.4f/%.4f"%(b[t,0,1],b[t,1,1]) for t in tt))

print()
print("Drift test over 20 time blocks (z-score of block mean vs overall mean, improved m2, m4; chi2/dof)")
for T in Ts:
    z=np.load("/work/prod_T%.1f.npz"%T)
    blocks = z['blocks']; K,nb,R,_ = blocks.shape
    pooled = blocks.transpose(1,0,2,3).reshape(nb,K*R,4)    # (nb, chains, 4)
    res=[]
    for col in (2,3):
        x = pooled[:,:,col]                      # (nb, chains)
        mb = x.mean(axis=1)                      # block means
        # error of block mean from chain scatter of block values (blocks within a chain are only weakly correlated)
        se = x.std(axis=1,ddof=1)/np.sqrt(x.shape[1])
        zsc = (mb - mb.mean())/se
        # trend: slope z-score using weighted least-squares
        t = np.arange(nb)-np.mean(np.arange(nb))
        slope = (t*(mb-mb.mean())/se**2).sum()/ (t*t/se**2).sum()
        slope_err = 1/np.sqrt((t*t/se**2).sum())
        res.append((np.sum(zsc**2)/(nb-1), slope/slope_err, zsc[0]))
    print("T=%.1f  m2: chi2/dof=%.2f slope-z=%.2f first-block-z=%.2f | m4: chi2/dof=%.2f slope-z=%.2f first-block-z=%.2f"%(T,*res[0],*res[1]))
