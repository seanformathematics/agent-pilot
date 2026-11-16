
import numpy as np
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
for T in Ts:
    d=np.load(f'/work/T{T}.npz'); a2=d['m2']; a4=d['m4']
    R=len(a2)
    U=1-a4.mean()/(3*a2.mean()**2)
    # jackknife
    S2=a2.sum(); S4=a4.sum()
    Uj=np.empty(R)
    for r in range(R):
        m2=(S2-a2[r])/(R-1); m4=(S4-a4[r])/(R-1)
        Uj[r]=1-m4/(3*m2**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    print(T, round(U,5), round(err,5))
