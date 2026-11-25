
import numpy as np
def U_of(S2,S4,n):
    return 1-(S4/n)/(3*(S2/n)**2)
out={}
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f"/work/T{T}.npz")
    S2,S4,n=d['S2'],d['S4'],int(d['n'])
    R=len(S2)
    U=U_of(S2.sum(),S4.sum(),n*R)
    # delete-one jackknife over replicas
    Uj=np.array([U_of(S2.sum()-S2[r],S4.sum()-S4[r],n*(R-1)) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    out[T]=(U,err,n*R)
    print(T, repr(U), repr(err), n, R)
