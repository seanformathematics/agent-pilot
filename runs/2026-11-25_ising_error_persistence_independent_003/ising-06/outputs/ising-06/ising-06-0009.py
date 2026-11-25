
import numpy as np
def U_of(S2,S4): return 1-np.mean(S4)/(3*np.mean(S2)**2)
def jk(S2,S4,nb=100):
    R=len(S2); idx=np.arange(R)%nb
    U=U_of(S2,S4)
    Us=np.array([U_of(S2[idx!=b],S4[idx!=b]) for b in range(nb)])
    err=np.sqrt((nb-1)/nb*np.sum((Us-Us.mean())**2))
    return U,err
for T,seed in [(1.9,201),(2.0,202),(2.1,203),(2.2,204),(2.3,205),(2.4,206)]:
    d=np.load(f'/work/sums_T{T}_s{seed}.npz')
    U,e=jk(d['S2'],d['S4'])
    # time-block check: U per block (pooled over replicas)
    Ub=[U_of(d['B2'][b],d['B4'][b]) for b in range(d['B2'].shape[0])]
    # first-half vs second-half drift
    print(f"T={T} U={U:.5f} err={e:.5f}  blocks:",np.round(Ub,4), " <|m|>=",round(d['SA'].mean(),4))
