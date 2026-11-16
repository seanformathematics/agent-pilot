import numpy as np
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def U_of(a2,a4): return 1-a4.mean()/(3*a2.mean()**2)
print("T  U  Uerr  (block-jackknife, 50 blocks)")
out={}
for T in Ts:
    d=np.load(f'/work/res_T{T:.2f}.npz'); a2=d['S2']; a4=d['S4']; R=len(a2)
    U=U_of(a2,a4)
    # block jackknife over 50 blocks of replicas
    nb=50; idx=np.arange(R)%nb
    Uj=np.array([U_of(a2[idx!=k],a4[idx!=k]) for k in range(nb)])
    err=np.sqrt((nb-1)/nb*np.sum((Uj-Uj.mean())**2))
    out[T]=(U,err)
    print(f"{T:.2f} {U:.5f} {err:.5f} n={int(d['n'])} R={R}")