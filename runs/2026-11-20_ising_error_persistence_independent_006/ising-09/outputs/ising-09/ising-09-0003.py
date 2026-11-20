import numpy as np, sys, time, json, subprocess
sys.path.insert(0,'/work')
from ising import run
def ustats(S2,S4):
    # per replica values; jackknife over replicas
    R=len(S2)
    def U(a2,a4): return 1 - a4.mean()/(3*a2.mean()**2)
    Ufull=U(S2,S4)
    idx=np.arange(R)
    # delete-one jackknife via sums (fast)
    t2=S2.sum(); t4=S4.sum()
    Uj=1-((t4-S4)/(R-1))/(3*((t2-S2)/(R-1))**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return Ufull, err
res={}
for T,seed in [(1.9,101),(2.0,102)]:
    t=time.time()
    m2,m4,m1,s=run(T,1000,2000,20000,seed)
    U,e=ustats(m2,m4)
    res[T]=dict(U=U,err=e,seed=seed,m2=float(m2.mean()),time=time.time()-t)
    np.savez(f'/work/res_{T}.npz',m2=m2,m4=m4,m1=m1)
    print(T,U,e,res[T]['m2'],time.time()-t,flush=True)