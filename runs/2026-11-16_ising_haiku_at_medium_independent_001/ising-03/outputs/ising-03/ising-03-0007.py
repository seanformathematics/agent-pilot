
lib = r'''
import numpy as np, json, time, os
N=16
def run(T, R, nsweep, ntherm, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    p4=np.exp(-4/T); p8=np.exp(-8/T)
    def sweep(s):
        for mk in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nb
            r=rng.random(s.shape)
            pa=np.where(dE==4,p4,np.where(dE==8,p8,1.0))
            flip=mk[None]&((dE<=0)|(r<pa))
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(ntherm): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R)
    for _ in range(nsweep):
        s=sweep(s)
        m=s.reshape(R,-1).sum(1,dtype=np.float64)/(N*N)
        m2=m*m
        S2+=m2; S4+=m2*m2
    return S2/nsweep, S4/nsweep
'''
open('/work/lib.py','w').write(lib)
import sys; sys.path.insert(0,'/work')
from lib import run
import json
for k,T in enumerate([1.5,1.6,1.7,1.8,1.9,2.0]):
    t=time.time()
    a2,a4=run(T,1000,20000,5000,1000+k)
    json.dump({'T':T,'m2':a2.tolist(),'m4':a4.tolist()},open(f'/work/res_{T}.json','w'))
    print('done',T,time.time()-t,flush=True)
    if time.time()-t>0 and k>=0 and False: break
