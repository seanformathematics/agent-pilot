
import numpy as np
R=512
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/T{T}.npz'); m2,m4=d['m2'],d['m4']
    u=1-m4.mean()/(3*m2.mean()**2)
    S2,S4=m2.sum(),m4.sum()
    a2=(S2-m2)/(R-1); a4=(S4-m4)/(R-1)
    jk=1-a4/(3*a2**2)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    print(T, repr(u), repr(err))
