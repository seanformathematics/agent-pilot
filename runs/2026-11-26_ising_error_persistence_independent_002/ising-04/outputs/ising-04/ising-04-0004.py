
import numpy as np
d=np.load('/work/T2.2_s2201.npz')
S2,S4,cnt=d['S2'],d['S4'],d['cnt']
R=S2.shape[1]
def U_of(idx_mask_blocks, reps):
    a2=S2[idx_mask_blocks][:,reps].sum(); a4=S4[idx_mask_blocks][:,reps].sum()
    n=cnt[idx_mask_blocks].sum()*len(reps)
    m2=a2/n; m4=a4/n
    return 1-m4/(3*m2*m2)
for b in range(S2.shape[0]):
    print(b, U_of([b], np.arange(R)))
def jack(blocks):
    reps=np.arange(R)
    U0=U_of(blocks,reps)
    Uj=np.array([U_of(blocks, np.delete(reps,r)) for r in range(R)])
    # jackknife variance
    var=(R-1)/R*np.sum((Uj-Uj.mean())**2)
    return U0, np.sqrt(var)
print('blocks 3-29:', jack(list(range(3,30))))
