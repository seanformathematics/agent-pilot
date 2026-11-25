
import sys, time, numpy as np
sys.path.insert(0,'/work')
from ising import run, binder_jk
t=time.time()
T=2.3
m2,m4,b2,b4=run(T,2000,8000,2000,seed=2301)
U,Ue=binder_jk(m2,m4)
np.savez('/work/res_T%.2f_s2301.npz'%T,m2=m2,m4=m4,b2=b2,b4=b4)
print("time",time.time()-t)
print("T",T,"U",U,"Uerr",Ue,"m2mean",m2.mean())
print("block U-ish m2 over time", b2.mean(axis=1).round(4))
