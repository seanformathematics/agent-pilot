
import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
def U_of(S2,S4,n):
    R=len(S2)
    m2=S2.sum()/(n*R); m4=S4.sum()/(n*R)
    return 1-m4/(3*m2*m2)
for T in Ts:
    d=np.load(f'/work/run_T{T}.npz')
    S2,S4=d['S2'],d['S4']; n=int(d['meas'])
    R=len(S2)
    full=U_of(S2,S4,n)
    out=[]
    for G in [10,20,50,100,250]:
        idx=np.array_split(np.arange(R),G)
        ests=[]
        for g in idx:
            mask=np.ones(R,bool); mask[g]=False
            ests.append(U_of(S2[mask],S4[mask],n))
        ests=np.array(ests); Gn=len(idx)
        err=np.sqrt((Gn-1)/Gn*np.sum((ests-ests.mean())**2))
        out.append((G,round(err,5)))
    print(T,round(full,5),out)
