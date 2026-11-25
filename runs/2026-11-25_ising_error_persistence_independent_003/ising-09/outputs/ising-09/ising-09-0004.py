
code = r'''
import numpy as np
N=16
def run(T, R, nsweep, nburn, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    p4=np.exp(-4/T); p8=np.exp(-8/T)
    S2=np.zeros(R); S4=np.zeros(R); cnt=0
    blk2=[]; blk4=[]; b2=np.zeros(R); b4=np.zeros(R); bs=0
    for sw in range(nsweep):
        for p in (0,1):
            nb=(np.roll(s,1,1).astype(np.int16)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nb
            prob=np.ones(s.shape)
            prob[dE==4]=p4; prob[dE==8]=p8
            u=rng.random(s.shape)
            flip=(u<prob)&masks[p]
            s=np.where(flip,-s,s).astype(np.int8)
        if sw>=nburn:
            m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
            m2=m*m
            S2+=m2; S4+=m2*m2; cnt+=1
            b2+=m2; b4+=m2*m2; bs+=1
            if bs==100:
                blk2.append(b2/bs); blk4.append(b4/bs); b2=np.zeros(R); b4=np.zeros(R); bs=0
    return S2/cnt, S4/cnt, np.array(blk2), np.array(blk4)

def binder_jk(m2,m4):
    # ratio estimator with jackknife over replicas
    R=len(m2)
    M2=m2.sum(); M4=m4.sum()
    U_full=1-(M4/R)/(3*(M2/R)**2)
    loo=np.empty(R)
    for k in range(R):
        a2=(M2-m2[k])/(R-1); a4=(M4-m4[k])/(R-1)
        loo[k]=1-a4/(3*a2*a2)
    var=(R-1)/R*np.sum((loo-loo.mean())**2)
    return U_full, np.sqrt(var)
'''
open('/work/ising.py','w').write(code)
import sys, time, numpy as np
sys.path.insert(0,'/work')
from ising import run, binder_jk
t=time.time()
T=1.9
m2,m4,b2,b4=run(T,2000,8000,2000,seed=1001)
U,Ue=binder_jk(m2,m4)
np.savez('/work/res_T%.2f_s1001.npz'%T,m2=m2,m4=m4,b2=b2,b4=b4)
# drift check: block averages of U over blocks
nb=b2.shape[0]
print("time",time.time()-t)
print("T",T,"U",U,"Uerr",Ue, "m2mean",m2.mean())
print("block m2 means over time", b2.mean(axis=1).round(4))
