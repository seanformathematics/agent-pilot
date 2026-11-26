
import sys, time, json
sys.path.insert(0,'/work')
import numpy as np
from ising import simulate
def jackknife_U(S2,S4,n):
    a=S2/n; b=S4/n
    R=len(a)
    A=a.sum(); B=b.sum()
    U=1-(B/R)/(3*(A/R)**2)
    Aj=A-a; Bj=B-b
    Uj=1-(Bj/(R-1))/(3*(Aj/(R-1))**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
res={}
seed_base=2024
out=[]
t0=time.time()
for k,T in enumerate([1.9,2.0,2.1,2.2,2.3,2.4]):
    seed=seed_base+k
    S2,S4,n,t2,t4=simulate(T,2000,seed,500,3000)
    U,err=jackknife_U(S2,S4,n)
    res[str(T)]=dict(U=U,err=err,n=n,seed=seed)
    print(T,U,err,n,time.time()-t0,flush=True)
    json.dump(res,open('/work/results.json','w'),indent=1)
