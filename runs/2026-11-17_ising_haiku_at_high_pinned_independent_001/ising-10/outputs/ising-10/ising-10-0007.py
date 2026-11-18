
import numpy as np
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
U_of=lambda a,b: 1-b/(3*a**2)
for T in Ts:
    d=np.load(f'/work/res_{T}.npz')
    m2r=d['m2b'].mean(axis=1); m4r=d['m4b'].mean(axis=1); R=len(m2r)
    U=U_of(m2r.mean(),m4r.mean()); S2=m2r.sum(); S4=m4r.sum()
    jk=np.array([U_of((S2-m2r[r])/(R-1),(S4-m4r[r])/(R-1)) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    print(T, repr(float(U)), repr(float(err)))
