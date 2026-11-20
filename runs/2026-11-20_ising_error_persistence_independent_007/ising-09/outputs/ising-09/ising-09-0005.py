import sys, numpy as np, time, json
sys.path.insert(0,'/work')
from ising import run
def ujk(m2,m4):
    R=len(m2)
    U=1-np.mean(m4)/(3*np.mean(m2)**2)
    tot2=m2.sum(); tot4=m4.sum()
    Uj=np.array([1-((tot4-m4[k])/(R-1))/(3*((tot2-m2[k])/(R-1))**2) for k in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
out={}
for T,seed in [(1.9,1001),(2.0,1002),(2.1,1003)]:
    t=time.time()
    m2,m4,m1=run(T,512,5000,20000,seed)
    U,e=ujk(m2,m4)
    np.savez(f'/work/res_{T}.npz',m2=m2,m4=m4,m1=m1)
    out[T]=(U,e)
    print(T,U,e,"m1",m1.mean(),"time",time.time()-t, flush=True)
