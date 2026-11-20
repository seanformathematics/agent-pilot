
import sys, time, numpy as np
sys.path.insert(0,'/work')
from ising import run
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
def ustat(r):
    m2=np.mean(r**2); m4=np.mean(r**4)
    return 1-m4/(3*m2**2)
def jk(r,nb=500):
    bl=np.array_split(r,nb)
    S2=sum(np.sum(b**2) for b in bl); S4=sum(np.sum(b**4) for b in bl); n=len(r)
    vals=[]
    for b in bl:
        n2=S2-np.sum(b**2); n4=S4-np.sum(b**4); nn=n-len(b)
        vals.append(1-(n4/nn)/(3*(n2/nn)**2))
    vals=np.array(vals)
    return np.sqrt((nb-1)/nb*np.sum((vals-vals.mean())**2))
t0=time.time()
out=[]
for k,T in enumerate(Ts):
    r=np.array(run(16,T,1000000,5000+k,2000))
    np.save(f'/work/prod_{T}.npy',r)
    U=ustat(r); e=jk(r)
    p=np.load(f'/work/pilot_{T}.npy')
    h1=ustat(r[:len(r)//2]); h2=ustat(r[len(r)//2:])
    print(T, 'U=%.5f err=%.5f'%(U,e), 'halves %.5f %.5f'%(h1,h2), 'pilot U=%.5f'%ustat(p), 'n=',len(r), 't=',round(time.time()-t0,1), flush=True)
