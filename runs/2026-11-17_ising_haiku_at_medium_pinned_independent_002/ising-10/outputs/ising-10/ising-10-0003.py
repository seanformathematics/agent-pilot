import sys, time, json, numpy as np
sys.path.insert(0,'/work')
from ising import run
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
V=256
res={}
t0=time.time()
for k,T in enumerate(Ts):
    o=np.array(run(T,1_000_000,5000,1000+k),dtype=np.float64)/V
    res[T]=o.tolist()
    np.save(f'/work/pilot_{T}.npy',o)
    if time.time()-t0>480: print('stopping at',T); break
print('elapsed',time.time()-t0)
def U_of(x):
    m2=np.mean(x**2); m4=np.mean(x**4)
    return 1-m4/(3*m2*m2)
for T in res:
    x=np.array(res[T])
    bins=x[:len(x)//1000*1000].reshape(-1,1000)
    nb=bins.shape[0]
    # jackknife over bins
    Ub=[]
    for b in range(nb):
        xx=np.delete(bins,b,axis=0).ravel()
        Ub.append(U_of(xx))
    Ub=np.array(Ub)
    err=np.sqrt((nb-1)/nb*np.sum((Ub-Ub.mean())**2))
    print(T, round(U_of(x),5), 'jk err', round(err,5))
