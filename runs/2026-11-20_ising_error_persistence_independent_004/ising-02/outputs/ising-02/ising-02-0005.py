
import numpy as np
def jk(S2,S4):
    R=len(S2)
    U=1-S4.mean()/(3*S2.mean()**2)
    tot2=S2.sum(); tot4=S4.sum()
    a2=(tot2-S2)/(R-1); a4=(tot4-S4)/(R-1)
    Ujk=1-a4/(3*a2*a2)
    err=np.sqrt((R-1)/R*np.sum((Ujk-Ujk.mean())**2))
    return U,err
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f"/work/T{T}_N16.npz")
    U,e=jk(d['S2'],d['S4'])
    print(T,repr(U),repr(e),int(d['n']))
