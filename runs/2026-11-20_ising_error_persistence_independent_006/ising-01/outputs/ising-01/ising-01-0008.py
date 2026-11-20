import numpy as np
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/T{T:.2f}_N16.npz')
    a2=d['S2'].mean(0); a4=d['S4'].mean(0); R=a2.size
    U=1-a4.mean()/(3*a2.mean()**2)
    tot2=a2.sum(); tot4=a4.sum()
    Uj=np.array([1-((tot4-a4[r])/(R-1))/(3*((tot2-a2[r])/(R-1))**2) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    print(T, repr(float(U)), repr(float(err)))
