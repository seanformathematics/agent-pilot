
import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
def Ufun(m2s,m4s,idx):
    a=m2s[:,idx].sum(); b=m4s[:,idx].sum()
    return None
res={}
for T in Ts:
    ms=np.load(f'/work/ms_T{T}_N16.npy').astype(float)  # (nmeas,R)
    m2=ms**2; m4=ms**4
    # per-replica sums
    S2=m2.sum(0); S4=m4.sum(0); n=ms.shape[0]
    R=ms.shape[1]
    U=1-(S4.sum()/n/R)/(3*(S2.sum()/n/R)**2)
    # jackknife over groups of replicas (50 groups of 20)
    G=50; gs=np.array_split(np.arange(R),G)
    Uj=[]
    for g in gs:
        keep=np.ones(R,bool); keep[g]=False
        a2=S2[keep].sum()/(n*keep.sum()); a4=S4[keep].sum()/(n*keep.sum())
        Uj.append(1-a4/(3*a2**2))
    Uj=np.array(Uj)
    err=np.sqrt((G-1)/G*np.sum((Uj-Uj.mean())**2))
    # bootstrap over replicas as check
    rng=np.random.default_rng(7)
    bs=[]
    for b in range(400):
        idx=rng.integers(0,R,R)
        a2=S2[idx].sum()/(n*R); a4=S4[idx].sum()/(n*R)
        bs.append(1-a4/(3*a2**2))
    # autocorrelation check of m^2 (mean over replicas)
    x=m2.mean(1)-m2.mean(); 
    ac=[np.mean(x[:-k]*x[k:])/np.var(x) for k in [1,5,10,20]]
    print(f"T={T} U={U:.5f} jack_err={err:.5f} boot_err={np.std(bs):.5f} ac(m2 lag 1,5,10,20 meas)={np.round(ac,3)}")
    res[T]=(U,err)
