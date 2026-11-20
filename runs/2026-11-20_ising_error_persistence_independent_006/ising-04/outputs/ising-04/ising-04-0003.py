code = r'''
import numpy as np, time, sys, json
N=16
def run(T, seed, R, n_eq, n_meas, block=20, out=None):
    rng=np.random.default_rng(seed)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    black=((ii+jj)%2==0)
    S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    acc_tab={d: np.exp(-d/T) for d in (4,8)}
    def sweep(S):
        for mask in (black,~black):
            nb=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)).astype(np.int16)
            dE=2*S*nb
            acc=np.where(dE>0, np.where(dE>4, acc_tab[8], acc_tab[4]), 1.0)
            flip=(rng.random(S.shape)<acc)&mask
            S=np.where(flip,-S,S).astype(np.int8)
        return S
    for k in range(n_eq):
        S=sweep(S)
    # measurement: block averages per replica
    nblk=n_meas//block
    m2b=np.zeros((nblk,R)); m4b=np.zeros((nblk,R))
    for b in range(nblk):
        a2=np.zeros(R); a4=np.zeros(R)
        for k in range(block):
            S=sweep(S)
            m=S.reshape(R,-1).mean(axis=1)
            a2+=m*m; a4+=m**4
        m2b[b]=a2/block; m4b[b]=a4/block
    return m2b, m4b, S

def binder(m2,m4):
    return 1-m4.mean()/(3*m2.mean()**2)

def jack_err(m2b,m4b):
    # replicas independent: jackknife over replicas (per-replica time averages)
    M2=m2b.mean(0); M4=m4b.mean(0); R=len(M2)
    U=binder(M2,M4)
    tot2=M2.sum(); tot4=M4.sum()
    Uj=np.empty(R)
    for r in range(R):
        a2=(tot2-M2[r])/(R-1); a4=(tot4-M4[r])/(R-1)
        Uj[r]=1-a4/(3*a2*a2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U, err
'''
open('/work/ising.py','w').write(code) if __import__('os').path.isdir('/work') else None
import os
os.makedirs('/work',exist_ok=True)
open('/work/ising.py','w').write(code)
import sys; sys.path.insert(0,'/work')
from ising import *
import time
t=time.time()
m2b,m4b,_=run(2.3, seed=101, R=1000, n_eq=1000, n_meas=6000, block=20)
print("time",time.time()-t)
# equilibration check: U vs block index in chunks
nb=m2b.shape[0]
for c in range(0,nb,30):
    sl=slice(c,c+30)
    print(c*20, round(binder(m2b[sl],m4b[sl]),5))
U,e=jack_err(m2b,m4b)
print("U",U,"err",e)
# tau estimate from replica-avg m2 series
s=m2b.mean(1); s=s-s.mean()
ac=[np.dot(s[:-k],s[k:])/np.dot(s,s) for k in range(1,30)]
print("acf of block m2 (block=20 sweeps):",np.round(ac[:10],3))
