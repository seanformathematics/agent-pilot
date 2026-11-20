
import numpy as np
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/T{T:.3f}.npz')
    m2=d['bm2'].mean(0); m4=d['bm4'].mean(0); R=len(m2)
    U=1-m4.mean()/(3*m2.mean()**2)
    S2=m2.sum(); S4=m4.sum()
    Uj=np.array([1-((S4-m4[r])/(R-1))/(3*((S2-m2[r])/(R-1))**2) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    print(f"{T} U={U:.6f} err={err:.6f}")
