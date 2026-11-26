
import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]; seeds={1.9:101,2.0:102,2.1:103,2.2:104,2.3:105,2.4:106}
def U_of(S2,S4):
    return 1-S4.mean()/(3*S2.mean()**2)
for T in Ts:
    d=np.load(f"/work/res_T{T:.3f}_s{seeds[T]}.npz")
    S2=d['S2']; S4=d['S4']; R=len(S2)
    U=U_of(S2,S4)
    jk=np.array([U_of(np.delete(S2,r),np.delete(S4,r)) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    print(T, repr(float(U)), repr(float(err)))
