
import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
def U_from(M):
    m2=(M**2).mean(axis=0); m4=(M**4).mean(axis=0)
    return 1-m4.mean()/(3*m2.mean()**2)
for T in Ts:
    M=np.load(f'/work/M_T{T}.npy')
    meas,R=M.shape
    U=U_from(M)
    m2c=(M**2).mean(axis=0); m4c=(M**4).mean(axis=0)
    S2=m2c.sum(); S4=m4c.sum()
    Ujk=np.empty(R)
    for r in range(R):
        a2=(S2-m2c[r])/(R-1); a4=(S4-m4c[r])/(R-1)
        Ujk[r]=1-a4/(3*a2**2)
    err=np.sqrt((R-1)/R*np.sum((Ujk-Ujk.mean())**2))
    U1=U_from(M[:meas//2]); U2=U_from(M[meas//2:])
    x=(M**2).mean(axis=1); x=x-x.mean()
    L=2000
    ac=np.correlate(x[:L],x[:L],'full')[L-1:]/(x.var()*L)
    tau=1+2*np.sum(ac[1:200][ac[1:200]>0])
    print(f"T={T} U={U:.5f} err_jk={err:.5f} halves {U1:.5f} {U2:.5f} tau_m2~{tau:.1f}")
