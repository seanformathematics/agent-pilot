
import numpy as np
def U_of(B2,B4):
    m2=B2.mean(); m4=B4.mean()
    return 1-m4/(3*m2**2)
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/data/T{T}.npz'); B2,B4,BA=d['B2'],d['B4'],d['BA']
    R=B2.shape[1]
    U=U_of(B2,B4)
    # delete-one jackknife over replicas
    S2=B2.sum(0); S4=B4.sum(0); nb=B2.shape[0]
    Uj=np.empty(R)
    tot2=S2.sum(); tot4=S4.sum()
    for r in range(R):
        m2=(tot2-S2[r])/((R-1)*nb); m4=(tot4-S4[r])/((R-1)*nb)
        Uj[r]=1-m4/(3*m2**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    # replica-level U spread check: err from 16 groups
    g=np.array_split(np.arange(R),16)
    Ug=[]
    for idx in g:
        mask=np.ones(R,bool); mask[idx]=False
        m2=B2[:,mask].mean(); m4=B4[:,mask].mean()
        Ug.append(1-m4/(3*m2**2))
    Ug=np.array(Ug); errg=np.sqrt(15/16*np.sum((Ug-Ug.mean())**2))
    print(f"T={T} U={U:.5f} err_jk={err:.5f} err_grp={errg:.5f}")
