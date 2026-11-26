
import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
for T in Ts:
    d=np.load(f"/work/data/T{T}.npz")
    M2=d["M2"]; M4=d["M4"]; R=len(M2)
    U=1-M4.mean()/(3*M2.mean()**2)
    # delete-group jackknife, groups of 10 replicas (100 groups)
    k=100; idx=np.arange(R).reshape(k,-1)
    Uj=[]
    for g in range(k):
        keep=np.ones(R,bool); keep[idx[g]]=False
        Uj.append(1-M4[keep].mean()/(3*M2[keep].mean()**2))
    Uj=np.array(Uj)
    err=np.sqrt((k-1)/k*np.sum((Uj-Uj.mean())**2))
    # per-replica bootstrap check
    rng=np.random.default_rng(0); bs=[]
    for b in range(1000):
        j=rng.integers(0,R,R)
        bs.append(1-M4[j].mean()/(3*M2[j].mean()**2))
    print(f"T={T} U={U:.5f} jackknife_err={err:.5f} bootstrap_err={np.std(bs):.5f} R={R}")
