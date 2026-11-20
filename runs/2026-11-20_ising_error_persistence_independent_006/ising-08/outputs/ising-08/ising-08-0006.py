
import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
for T in Ts:
    d=np.load(f'/work/T{T:.3f}.npz')
    m2=d['bm2'].mean(0); m4=d['bm4'].mean(0)   # per-replica time averages, shape R
    R=len(m2)
    U=1-m4.mean()/(3*m2.mean()**2)
    # jackknife over replicas
    S2=m2.sum(); S4=m4.sum()
    Uj=np.empty(R)
    for r in range(R):
        a2=(S2-m2[r])/(R-1); a4=(S4-m4[r])/(R-1)
        Uj[r]=1-a4/(3*a2**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    # autocorrelation check of block means of m2 (replica-averaged)
    x=d['bm2'].mean(1); x=x-x.mean()
    ac1=np.dot(x[:-1],x[1:])/np.dot(x,x)
    # per-replica U spread for reference
    # alternative: split replicas into 10 groups
    g=np.array_split(np.arange(R),10)
    Ug=[1-m4[idx].mean()/(3*m2[idx].mean()**2) for idx in g]
    print(f"T={T} U={U:.5f} jackErr={err:.5f} groupStd/sqrt10={np.std(Ug,ddof=1)/np.sqrt(10):.5f} lag1ac(blocks)={ac1:.3f} <m2>={m2.mean():.4f}")
