
import numpy as np
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/raw_T{T}.npz'); m2,m4=d['m2'],d['m4']; R=len(m2)
    U=1-m4.mean()/(3*m2.mean()**2)
    S2=m2.sum(); S4=m4.sum()
    Uj=np.array([1-((S4-m4[i])/(R-1))/(3*((S2-m2[i])/(R-1))**2) for i in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    print(T, repr(float(U)), repr(float(err)))
