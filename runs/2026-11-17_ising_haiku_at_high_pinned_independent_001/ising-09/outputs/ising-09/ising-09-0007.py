
import numpy as np
def U_jack(m2r,m4r):
    R=len(m2r)
    U=1-m4r.mean()/(3*m2r.mean()**2)
    S2=m2r.sum(); S4=m4r.sum()
    a2=(S2-m2r)/(R-1); a4=(S4-m4r)/(R-1)
    Uj=1-a4/(3*a2*a2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
for T in [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]:
    d=np.load(f'/work/res/T{T}.npz')
    U,e=U_jack(d['m2'],d['m4'])
    print(T, repr(float(U)), repr(float(e)))
