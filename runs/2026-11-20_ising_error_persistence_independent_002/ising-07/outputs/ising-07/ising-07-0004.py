
import sys, time, json, numpy as np
sys.path.insert(0,'/work')
from ising import run, N
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
results={}
for T in Ts:
    t=time.time()
    seed=int(7919*T*10)+12345
    s,out=run(T,1000,10000,seed=seed,rec_every=1)
    o=out[2000:]
    m2=o[:,:,0].mean(axis=0); m4=o[:,:,1].mean(axis=0)
    U=1-m4.mean()/(3*m2.mean()**2)
    g=10; G=1000//g
    m2g=m2.reshape(G,g).mean(1); m4g=m4.reshape(G,g).mean(1)
    S2=m2g.sum(); S4=m4g.sum()
    Uj=np.array([1-((S4-m4g[k])/(G-1))/(3*((S2-m2g[k])/(G-1))**2) for k in range(G)])
    err=np.sqrt((G-1)/G*np.sum((Uj-Uj.mean())**2))
    # also per-block consistency: split time into 4 chunks
    chunks=[]
    for c in range(4):
        oc=o[c*2000:(c+1)*2000]
        mm2=oc[:,:,0].mean(0); mm4=oc[:,:,1].mean(0)
        chunks.append(1-mm4.mean()/(3*mm2.mean()**2))
    results[T]=dict(seed=seed,U=float(U),err=float(err),chunks=[float(x) for x in chunks])
    print(f"T={T} seed={seed} U={U:.5f} err={err:.5f} chunk U={np.round(chunks,4)} t={time.time()-t:.0f}s",flush=True)
json.dump(results,open('/work/results_prod.json','w'),indent=1)
