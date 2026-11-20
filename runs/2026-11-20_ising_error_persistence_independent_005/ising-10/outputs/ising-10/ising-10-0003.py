
code = r'''
import numpy as np, time, sys, os
N=16; R=1000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def sweep(s,beta,rng):
    for p in (0,1):
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*h
        acc=np.exp(-beta*np.clip(dE,0,None))
        flip=(rng.random(s.shape)<acc)&masks[p]
        s[flip]*=-1
    return s
def run(T,seed,nburn,nmeas):
    rng=np.random.default_rng(seed)
    beta=1/T
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N)).astype(np.int8)
    for k in range(nburn): sweep(s,beta,rng)
    m2s=np.zeros(R); m4s=np.zeros(R); absm=np.zeros(R)
    for k in range(nmeas):
        sweep(s,beta,rng)
        m=s.reshape(R,-1).mean(axis=1)
        m2=m*m
        m2s+=m2; m4s+=m2*m2; absm+=np.abs(m)
    np.savez(f'/work/acc_T{T}_s{seed}.npz',m2=m2s,m4=m4s,absm=absm,nmeas=nmeas)
def analyze(fn):
    d=np.load(fn); n=int(d['nmeas'])
    m2=d['m2']/n; m4=d['m4']/n
    U_r=1-m4/(3*m2**2)
    Ubar=1-m4.mean()/(3*m2.mean()**2)
    # replica jackknife
    Rr=len(m2); S2=m2.sum(); S4=m4.sum()
    jk=np.empty(Rr)
    for i in range(Rr):
        a2=(S2-m2[i])/(Rr-1); a4=(S4-m4[i])/(Rr-1)
        jk[i]=1-a4/(3*a2**2)
    err=np.sqrt((Rr-1)/Rr*np.sum((jk-jk.mean())**2))
    return Ubar, err, U_r.std(ddof=1), np.mean(d['absm']/n)
'''
open('/work/ising_lib.py','w').write(code)
import sys; sys.path.insert(0,'/work')
import importlib, time
import ising_lib as L
for fn in ['/work/acc_T1.9_s101.npz','/work/acc_T2.0_s102.npz']:
    print(fn, L.analyze(fn))
for T,seed in [(2.1,103),(2.2,104)]:
    t=time.time(); L.run(T,seed,2000,15000)
    print(T, L.analyze(f'/work/acc_T{T}_s{seed}.npz'), time.time()-t)
