
import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
for T in Ts:
    d=np.load(f'/work/raw_T{T}.npz')
    m2,m4=d['m2'],d['m4']; R=len(m2)
    U=1-m4.mean()/(3*m2.mean()**2)
    # jackknife over replicas
    S2=m2.sum(); S4=m4.sum()
    Uj=np.array([1-((S4-m4[i])/(R-1))/(3*((S2-m2[i])/(R-1))**2) for i in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    # per-replica spread check: split replicas into 5 groups
    g=np.array_split(np.arange(R),10)
    Ug=[1-m4[idx].mean()/(3*m2[idx].mean()**2) for idx in g]
    print(f"T={T} U={U:.5f} jackknife_err={err:.5f} groups10_err={np.std(Ug,ddof=1)/np.sqrt(10):.5f}")
