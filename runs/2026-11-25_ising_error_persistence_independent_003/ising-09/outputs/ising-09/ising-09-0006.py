
import sys, time, numpy as np
sys.path.insert(0,'/work')
from ising import run, binder_jk
out={}
for k,T in enumerate([2.0,2.1,2.2,2.4]):
    t=time.time()
    seed=3000+k*100+int(round(T*10))
    m2,m4,b2,b4=run(T,2000,8000,2000,seed=seed)
    U,Ue=binder_jk(m2,m4)
    np.savez('/work/res_T%.2f_s%d.npz'%(T,seed),m2=m2,m4=m4,b2=b2,b4=b4)
    print("T",T,"seed",seed,"U",U,"Uerr",Ue,"m2",m2.mean(),"time",round(time.time()-t,1), flush=True)
