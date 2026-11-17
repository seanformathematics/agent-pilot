
import numpy as np, json, time
exec(open('/work/ising.py').read())
def U_of(m2,m4): return 1-np.mean(m4)/(3*np.mean(m2)**2)
def jkerr(m2,m4):
    R=len(m2); jk=np.array([U_of(np.delete(m2,i),np.delete(m4,i)) for i in range(R)])
    return U_of(m2,m4), np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
results={}
for k,T in enumerate([1.5,1.6,1.7,1.8,1.9]):
    t0=time.time()
    seed=2000+k
    m2,m4,am,series=run(T, seed=seed, R=400, nequil=2000, nmeas=10000)
    U,e=jkerr(m2,m4)
    # halves check: recompute from series
    x=series.astype(float)
    h=x.shape[0]//2
    m2a=(x[:h]**2).mean(0); m4a=(x[:h]**4).mean(0)
    m2b=(x[h:]**2).mean(0); m4b=(x[h:]**4).mean(0)
    Ua=U_of(m2a,m4a); Ub=U_of(m2b,m4b)
    results[T]=dict(U=float(U),err=float(e),seed=seed,Uhalf1=float(Ua),Uhalf2=float(Ub))
    print(T, round(U,5), round(e,5), 'halves', round(Ua,5), round(Ub,5), 'time', round(time.time()-t0,1), flush=True)
json.dump(results,open('/work/res_batch1.json','w'),indent=1)
