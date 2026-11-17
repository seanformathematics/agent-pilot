
import sys; sys.path.insert(0,'/work')
import numpy as np, time
from swlib import *
d=np.load('/work/pilot.npz')
ts2=d['ts_m2']; ts4=d['ts_m4']
# drift check: compare first and last quarter of pilot time series (group-averaged m^2)
q=len(ts2)//4
for k,T in enumerate(Ts):
    a=ts2[:q,k].mean(); b=ts2[-q:,k].mean()
    print("pilot T=%.1f first-quarter<m2>=%.4f last-quarter=%.4f"%(T,a,b))
# production with new seed
seed=777
rng=np.random.default_rng(seed)
reps=150
T_rep=np.repeat(Ts,reps); R=len(T_rep)
p,U,V,rep=setup(T_rep)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=R*NN)
t0=time.time()
for it in range(800):
    s=sw_sweep(s,rng,p,U,V,rep,R)
S2=np.zeros(R); S4=np.zeros(R); n=0
meas=4000
ts2=np.zeros((meas,14)); ts4=np.zeros((meas,14))
for it in range(meas):
    s=sw_sweep(s,rng,p,U,V,rep,R)
    m=s.reshape(R,NN).sum(1)/NN
    m2=m*m; m4=m2*m2
    S2+=m2; S4+=m4; n+=1
    ts2[it]=m2.reshape(14,reps).mean(1); ts4[it]=m4.reshape(14,reps).mean(1)
    if time.time()-t0>480: 
        print("stopped early at",it); break
print("n meas",n,"time",time.time()-t0)
groups=[np.arange(k*reps,(k+1)*reps) for k in range(14)]
res=jk_U(S2/n,S4/n,groups)
for T,(u,e) in zip(Ts,res):
    print(T, round(u,5), round(e,5))
np.savez('/work/prod.npz',S2=S2,S4=S4,n=n,ts2=ts2,ts4=ts4,seed=seed)
