
import numpy as np
R=1000
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f"/work/rep_T{T}.npz"); m2=d['m2']; m4=d['m4']
    U=1-m4.mean()/(3*m2.mean()**2)
    Uj=np.array([1-np.delete(m4,r).mean()/(3*np.delete(m2,r).mean()**2) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    print(T, repr(float(U)), repr(float(err)))
