import numpy as np
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def U(a,b): return 1-b.mean()/(3*a.mean()**2)
for T in Ts:
    d=np.load(f'/work/res/T{T}_N16_b.npz'); m2=d['m2']; m4=d['m4']; R=len(m2)
    full=U(m2,m4)
    loo=np.array([U(np.delete(m2,k),np.delete(m4,k)) for k in range(R)])
    err=np.sqrt((R-1)/R*np.sum((loo-loo.mean())**2))
    print(f"{T} {full:.7f} {err:.7f}")