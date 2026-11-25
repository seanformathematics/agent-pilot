
import numpy as np
def jk(A,B):
    R=A.shape[0]
    Ar=A.mean(1); Br=B.mean(1)
    sa=Ar.sum(); sb=Br.sum()
    a=(sa-Ar)/(R-1); b=(sb-Br)/(R-1)
    Us=1-b/(3*a*a)
    U=1-(sb/R)/(3*(sa/R)**2)
    err=np.sqrt((R-1)/R*np.sum((Us-Us.mean())**2))
    return U,err
for T in ['1.900','2.000','2.100','2.200','2.300','2.400']:
    d=np.load(f'/work/state_T{T}.npz')
    U,e=jk(d['A'],d['B'])
    print(T, repr(float(U)), repr(float(e)))
