
import sys, time
sys.path.insert(0,'/work')
import numpy as np
from metro import MetroBatch

N=4; Ns=16
states = np.arange(2**Ns, dtype=np.uint32)
bits = ((states[:,None] >> np.arange(Ns, dtype=np.uint32)[None,:]) & 1).astype(np.int8)*2-1
S = bits.reshape(-1,N,N)
E = -(S*np.roll(S,-1,axis=2)).sum(axis=(1,2)) - (S*np.roll(S,-1,axis=1)).sum(axis=(1,2))
m = S.sum(axis=(1,2)).astype(np.float64)/Ns
def exact(T):
    w=np.exp(-(E-E.min())/T); w/=w.sum()
    m2=(w*m**2).sum(); m4=(w*m**4).sum()
    return m2,m4,1-m4/(3*m2**2),(w*E).sum()/Ns

def energy(s):
    return -(s*np.roll(s,-1,axis=2)).sum(axis=(1,2),dtype=np.int64) - (s*np.roll(s,-1,axis=1)).sum(axis=(1,2),dtype=np.int64)

t0=time.time()
zs=[]
for T in (2.5,2.0,3.5):
    ex=exact(T)
    for rep in range(6 if T==2.5 else 3):
        R=500; sim=MetroBatch(4,R,T,seed=70000+int(T*10)*10+rep,start=np.arange(R)%2)
        for _ in range(200): sim.sweep()
        Ms=np.empty((4000,R)); Es=np.empty((4000,R))
        for t in range(4000):
            Ms[t]=sim.sweep(); Es[t]=energy(sim.s)
        mm=Ms/Ns
        c2=(mm**2).mean(axis=0); c4=(mm**4).mean(axis=0); n=R
        U=1-c4.mean()/(3*c2.mean()**2)
        Ul=1-((c4.sum()-c4)/(n-1))/(3*((c2.sum()-c2)/(n-1))**2)
        err=np.sqrt((n-1)/n*((Ul-Ul.mean())**2).sum())
        ce=Es.mean(axis=0)/Ns; eerr=ce.std(ddof=1)/np.sqrt(n)
        zs.append((U-ex[2])/err)
        print("T=%.1f rep %d: U=%.5f+-%.5f exact %.5f (z=%.2f) | E/N^2=%.5f+-%.5f exact %.5f (z=%.2f)  [%.0fs]"%(T,rep,U,err,ex[2],(U-ex[2])/err,ce.mean(),eerr,ex[3],(ce.mean()-ex[3])/eerr,time.time()-t0),flush=True)
zs=np.array(zs); print("z mean %.2f rms %.2f (n=%d)"%(zs.mean(),np.sqrt((zs**2).mean()),len(zs)))
