
import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
def U_of(m2,m4): return 1-np.mean(m4)/(3*np.mean(m2)**2)
for T in Ts:
    d=np.load(f'/work/T{T}.npz')
    m2=d['m2']; m4=d['m4']; R=len(m2)
    U=U_of(m2,m4)
    # jackknife over replicas
    S2=m2.sum(); S4=m4.sum()
    Uj=np.array([1-((S4-m4[i])/(R-1))/(3*((S2-m2[i])/(R-1))**2) for i in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    # block check: 10 blocks of replicas
    bl=[U_of(b2,b4) for b2,b4 in zip(np.array_split(m2,10),np.array_split(m4,10))]
    print(T, round(U,5), 'jk_err',round(err,5), 'blockstd/sqrt10', round(np.std(bl,ddof=1)/np.sqrt(10),5), 'cnt',int(d['cnt']))
