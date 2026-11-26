
import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
def U_of(S2,S4):
    return 1-S4.mean()/(3*S2.mean()**2)
for T in Ts:
    d=np.load(f'/work/T{T}_N16.npz')
    S2=d['S2'];S4=d['S4']; R=S2.shape[1]
    # per-replica time averages
    a2=S2.mean(0); a4=S4.mean(0)
    U=U_of(S2,S4)
    G=50; idx=np.arange(R)%G
    Uj=[]
    for g in range(G):
        k=idx!=g
        Uj.append(1-a4[k].mean()/(3*a2[k].mean()**2))
    Uj=np.array(Uj)
    err=np.sqrt((G-1)/G*np.sum((Uj-Uj.mean())**2))
    # also block-level jackknife over time blocks as cross-check
    nb=S2.shape[0]
    Ub=[]
    for b in range(nb):
        k=np.arange(nb)!=b
        Ub.append(1-S4[k].mean()/(3*S2[k].mean()**2))
    Ub=np.array(Ub)
    errb=np.sqrt((nb-1)/nb*np.sum((Ub-Ub.mean())**2))
    print(T, round(U,5), "jk-replica err", round(err,5), "jk-time err", round(errb,5))
