import numpy as np, sys, time
sys.path.insert(0,'/work')
from ising import run
def ustats(S2,S4):
    R=len(S2)
    Ufull=1 - S4.mean()/(3*S2.mean()**2)
    t2=S2.sum(); t4=S4.sum()
    Uj=1-((t4-S4)/(R-1))/(3*((t2-S2)/(R-1))**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return Ufull, err
for T,seed in [(2.1,103),(2.2,104)]:
    t=time.time()
    m2,m4,m1,s=run(T,1000,2000,20000,seed)
    U,e=ustats(m2,m4)
    np.savez(f'/work/res_{T}.npz',m2=m2,m4=m4,m1=m1)
    print(T,U,e,m2.mean(),time.time()-t,flush=True)