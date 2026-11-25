
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
code = r'''
import numpy as np, os, sys, time
N=16
def make_masks():
    i,j=np.indices((N,N))
    return ((i+j)%2==0), ((i+j)%2==1)
def run(T, R, seed, nsweep, S=None, rng=None, meas=True):
    if rng is None: rng=np.random.default_rng(seed)
    if S is None: S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ev,od=make_masks()
    acc=np.ones(9)
    for d in range(1,5):
        acc[d+4]=np.exp(-2.0*d/T)
    acc[:4]=0.0  # not used for negative delta
    out=[]
    for sw in range(nsweep):
        for mask in (ev,od):
            nb=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2))
            delta=(S*nb).astype(np.int8)  # in -4..4
            p=np.where(delta<=0,1.0,acc[np.clip(delta+4,0,8)])
            flip=mask & (rng.random((R,N,N))<p)
            S=np.where(flip,-S,S).astype(np.int8)
        if meas:
            m=S.reshape(R,-1).mean(axis=1)
            out.append(m)
    return S, rng, np.array(out) if meas else None
'''
open('/work/ising.py','w').write(code)
import sys; sys.path.insert(0,'/work')
from ising import *
t=time.time()
S,rng,ms=run(2.27,1000,1,50)
print("time per sweep (1000 reps):",(time.time()-t)/50)
m2=(ms**2).mean(); m4=(ms**4).mean()
print("U",1-m4/(3*m2**2), m2)
