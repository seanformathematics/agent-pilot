
import sys, time, json
sys.path.insert(0,'/work')
import numpy as np
from ising import run, binder_jk
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
res={}
t0=time.time()
for k,T in enumerate(Ts):
    seed=1000+k
    m2,m4=run(T,1000,seed,3000,4000)
    U,err=binder_jk(m2,m4)
    res[str(T)]=dict(U=float(U),err=float(err),seed=seed,m2=float(m2.mean()),m4=float(m4.mean()))
    print(T,U,err,flush=True)
    np.save(f'/work/m2_{T}.npy',m2); np.save(f'/work/m4_{T}.npy',m4)
json.dump(res,open('/work/res_main.json','w'),indent=1)
print("total time",time.time()-t0)
