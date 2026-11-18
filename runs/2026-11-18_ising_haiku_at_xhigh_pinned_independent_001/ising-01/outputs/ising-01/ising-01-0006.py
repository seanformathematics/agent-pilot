
import sys, time
sys.path.insert(0,'/work')
import numpy as np
import ising_sw as M

T=2.8
Us=[];Es=[]
t=time.time()
for seed in range(101,113):
    m2b,m4b=M.run(T,seed=seed,R=256,burn=300,L=1000,block=10)
    U,e=M.jackknife_U(m2b.mean(0),m4b.mean(0))
    Us.append(U);Es.append(e)
    print(seed, round(U,5), round(e,5), round(time.time()-t,1), flush=True)
Us=np.array(Us);Es=np.array(Es)
print("mean U",Us.mean(),"std across runs",Us.std(ddof=1),"mean jackknife err",Es.mean(), "std err of mean", Us.std(ddof=1)/np.sqrt(len(Us)))
np.save('/work/rep_T2.8.npy',Us)
