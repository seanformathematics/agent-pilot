
import sys, time, numpy as np, json
sys.path.insert(0,'/work')
from wolff import wolff_run
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
res={}
def U_of(a,b): return 1-b.mean()/(3*a.mean()**2)
for k,T in enumerate(Ts):
    seed=1000+k
    t=time.time()
    m2,m4=wolff_run(16,T,1000000,seed,nburn=5000)
    np.savez(f'/work/prod_T{T}.npz', m2=m2, m4=m4)
    nb=100; L=len(m2)//nb
    us=np.array([U_of(m2[b*L:(b+1)*L],m4[b*L:(b+1)*L]) for b in range(nb)])
    err=us.std(ddof=1)/np.sqrt(nb)
    res[T]=(U_of(m2,m4),err,seed)
    print(T, seed, "U=%.6f err=%.6f  <m2>=%.5f time=%.1f"%(U_of(m2,m4),err,m2.mean(),time.time()-t), flush=True)
json.dump({str(k):v for k,v in res.items()}, open('/work/prod_res.json','w'))
