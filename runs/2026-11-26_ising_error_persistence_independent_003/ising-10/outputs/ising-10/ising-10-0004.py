import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
def U_of(m2,m4):
    return 1-m4.mean()/(3*m2.mean()**2)
for T in Ts:
    d=np.load(f'/work/T{T}.npz'); m2=d['m2'].mean(0); m4=d['m4'].mean(0)  # per replica time averages
    R=len(m2)
    U=U_of(m2,m4)
    # jackknife over replicas
    idx=np.arange(R)
    Uj=np.array([U_of(np.delete(m2,i),np.delete(m4,i)) for i in idx])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    # block-wise drift check: U from first half vs second half of blocks
    m2b=d['m2']; m4b=d['m4']
    Ua=U_of(m2b[:20].mean(0),m4b[:20].mean(0)); Ub=U_of(m2b[20:].mean(0),m4b[20:].mean(0))
    print(f"T={T} U={U:.5f} err={err:.5f}  U(first half)={Ua:.5f} U(second half)={Ub:.5f} <|m|>={d['m1'].mean():.4f}")