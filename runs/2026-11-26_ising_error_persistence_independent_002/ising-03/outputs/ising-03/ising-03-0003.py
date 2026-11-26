code = r'''
import numpy as np, time
N=16
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
MASKS=[((ii+jj)%2==p) for p in (0,1)]

def sweeps_run(s,rng,T,nsweep,record=None):
    sh=np.array([-4,-2,0,2,4])
    acc=np.where(sh<=0,1.0,np.exp(-2.0*sh/T)).astype(np.float32)
    B=s.shape[0]
    for sw in range(nsweep):
        for p in (0,1):
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            idx=((s*nb)+4)//2
            r=rng.random((B,N,N),dtype=np.float32)
            flip=(r<acc[idx])&MASKS[p]
            s=np.where(flip,-s,s).astype(np.int8)
        if record is not None:
            record(sw,s)
    return s

def run_T(T,B,nburn,nmeas,seed):
    rng=np.random.Generator(np.random.PCG64(seed))
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(B,N,N))
    s=sweeps_run(s,rng,T,nburn)
    S2=np.zeros(B); S4=np.zeros(B)
    mser=[]
    def rec(sw,s):
        m=s.reshape(B,-1).mean(axis=1,dtype=np.float64)
        S2[:]+=m*m; S4[:]+=m**4
        mser.append(m.astype(np.float32))
    s=sweeps_run(s,rng,T,nmeas,record=rec)
    return S2,S4,nmeas,np.array(mser)

def binder_jk(S2,S4,cnt):
    B=len(S2)
    tot2=S2.sum(); tot4=S4.sum(); C=cnt*B
    U=1-(tot4/C)/(3*(tot2/C)**2)
    t2=tot2-S2; t4=tot4-S4; Cb=cnt*(B-1)
    Uj=1-(t4/Cb)/(3*(t2/Cb)**2)
    err=np.sqrt((B-1)/B*np.sum((Uj-Uj.mean())**2))
    return U,err

def tau_int(x,maxlag=300):
    x=x-x.mean(); v=np.mean(x*x)
    tau=0.5
    for l in range(1,maxlag):
        c=np.mean(x[:-l]*x[l:])/v
        if c<0.02: break
        tau+=c
    return tau
'''
open('/work/ising.py','w').write(code)
import sys; sys.path.insert(0,'/work')
from ising import *
t0=time.time()
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    S2,S4,cnt,mser=run_T(T,B=500,nburn=1000,nmeas=1500,seed=int(T*1000))
    U,err=binder_jk(S2,S4,cnt)
    taus=[tau_int(mser[:,b].astype(float)**2) for b in range(20)]
    print(f"T={T} U={U:.5f} err={err:.5f} tau_m2~{np.mean(taus):.1f} time={time.time()-t0:.1f}",flush=True)
