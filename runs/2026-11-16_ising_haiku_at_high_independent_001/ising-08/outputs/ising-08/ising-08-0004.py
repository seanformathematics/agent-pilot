
import numpy as np, json, time
exec(open('/work/ising.py').read())
def U_of(m2,m4): return 1-np.mean(m4)/(3*np.mean(m2)**2)
def jkerr(m2,m4):
    R=len(m2); jk=np.array([U_of(np.delete(m2,i),np.delete(m4,i)) for i in range(R)])
    return U_of(m2,m4), np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
results={}
for k,T in enumerate([2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]):
    t0=time.time()
    seed=3000+k
    m2,m4,am,series=run(T, seed=seed, R=400, nequil=2000, nmeas=10000)
    U,e=jkerr(m2,m4)
    x=series.astype(float); h=x.shape[0]//2
    Ua=U_of((x[:h]**2).mean(0),(x[:h]**4).mean(0)); Ub=U_of((x[h:]**2).mean(0),(x[h:]**4).mean(0))
    results[T]=dict(U=float(U),err=float(e),seed=seed,Uhalf1=float(Ua),Uhalf2=float(Ub))
    print(T, '%.6f'%U, '%.6f'%e, 'halves %.5f %.5f'%(Ua,Ub), 'time', round(time.time()-t0,1), flush=True)
json.dump(results,open('/work/res_batch2.json','w'),indent=1)
