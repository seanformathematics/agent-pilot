
import numpy as np
def U_of(m2,m4): return 1-m4.mean()/(3*m2.mean()**2)
def binder_jk(m2b,m4b):
    m2=m2b.mean(0); m4=m4b.mean(0)
    R=len(m2)
    U0=U_of(m2,m4)
    jk=np.array([U_of(np.delete(m2,i),np.delete(m4,i)) for i in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    return U0,err
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f"/work/T{T:.2f}.npz"); m2b,m4b=d['m2b'],d['m4b']
    U0,e=binder_jk(m2b,m4b)
    h1=binder_jk(m2b[:50],m4b[:50])[0]; h2=binder_jk(m2b[50:],m4b[50:])[0]
    # error using 10-block chunks of 10 blocks each per replica (coarser, checks correlations)
    m2c=m2b.reshape(10,10,-1).mean(1); m4c=m4b.reshape(10,10,-1).mean(1)
    # treat (replica, chunk) as independent units
    Uc,ec=binder_jk(m2c.reshape(10,-1)[:, :],m4c.reshape(10,-1)) if False else (None,None)
    print(f"T={T} U={U0:.5f} err={e:.2e} halves {h1:.5f} {h2:.5f} <m^2>={m2b.mean():.4f}")
