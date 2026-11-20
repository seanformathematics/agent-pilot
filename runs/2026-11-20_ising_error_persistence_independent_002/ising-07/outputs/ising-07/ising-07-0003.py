
import sys, time, numpy as np
sys.path.insert(0,'/work')
from ising import run, N
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
res={}
t0=time.time()
for T in Ts:
    t=time.time()
    s,out=run(T,1000,4000,seed=int(1000*T),rec_every=1)
    # out shape (4000,R,2)
    blk=out.reshape(40,100,1000,2).mean(axis=1)  # (40,R,2)
    m2b=blk[:,:,0].mean(axis=1); 
    Ub=1-blk[:,:,1].mean(axis=1)/(3*blk[:,:,0].mean(axis=1)**2)
    # discard first 15 blocks
    o=out[1500:]
    m2=o[:,:,0].mean(axis=0); m4=o[:,:,1].mean(axis=0)  # per replica
    U=1-m4.mean()/(3*m2.mean()**2)
    # jackknife over replicas (groups of 10)
    g=10; G=1000//g
    m2g=m2.reshape(G,g).mean(1); m4g=m4.reshape(G,g).mean(1)
    # jackknife
    S2=m2g.sum(); S4=m4g.sum()
    Uj=np.array([1-((S4-m4g[k])/(G-1))/(3*((S2-m2g[k])/(G-1))**2) for k in range(G)])
    err=np.sqrt((G-1)/G*np.sum((Uj-Uj.mean())**2))
    print(f"T={T} U={U:.5f} err={err:.5f}  m2 block means(last5)={np.round(m2b[-5:],4)} time={time.time()-t:.1f}s")
    print("   U per 100-sweep block (first 10 / last 10 mean):", np.round(Ub[:10],3), np.round(Ub[-10:].mean(),4))
print("total",time.time()-t0)
