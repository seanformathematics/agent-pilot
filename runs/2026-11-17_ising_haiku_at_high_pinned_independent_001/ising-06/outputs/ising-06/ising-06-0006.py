import numpy as np
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def U_of(m2,m4): return 1-m4.mean()/(3*m2.mean()**2)
for T in Ts:
    d=np.load(f'/work/res_N16_T{T:.2f}.npz')
    m2,m4=d['m2'],d['m4']; R=len(m2)
    U=U_of(m2,m4); nb=50; idx=np.arange(R)%nb
    Uj=np.array([U_of(m2[idx!=b],m4[idx!=b]) for b in range(nb)])
    err=np.sqrt((nb-1)/nb*np.sum((Uj-Uj.mean())**2))
    print(f"{T:.2f} {U:.6f} {err:.7f} {int(d['seed'])}")
