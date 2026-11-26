
import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
for T in Ts:
    ms=np.load(f'/work/ms_T{T}_N16.npy').astype(np.float64)
    n,R=ms.shape
    S2=(ms**2).sum(0); S4=(ms**4).sum(0)
    U=1-(S4.sum()/(n*R))/(3*(S2.sum()/(n*R))**2)
    G=50; gs=np.array_split(np.arange(R),G); Uj=[]
    for g in gs:
        keep=np.ones(R,bool); keep[g]=False
        a2=S2[keep].sum()/(n*keep.sum()); a4=S4[keep].sum()/(n*keep.sum())
        Uj.append(1-a4/(3*a2**2))
    Uj=np.array(Uj)
    err=np.sqrt((G-1)/G*np.sum((Uj-Uj.mean())**2))
    print(f"T={T} N_meas={n*R} U={U:.6f} U_err={err:.2e} min_m={ms.min():.3f}")
