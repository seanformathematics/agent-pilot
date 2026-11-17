
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16; per=100; Ts=np.linspace(1.5,2.8,14)
R=per*len(Ts)
rng=np.random.default_rng(20240601)
Tr=np.repeat(Ts,per)
beta=(1/Tr)[:,None,None]
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s):
    for m in masks:
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*h
        acc=np.logical_or(np.less_equal(dE,0), rng.random(s.shape)<np.exp(-beta*np.maximum(dE,0)))
        flip=np.logical_and(acc,m)
        s=np.where(flip,-s,s).astype(np.int8)
    return s
burn=3000; total=30000; every=5
t0=time.time()
for _ in range(burn): s=sweep(s)
M=[]
for k in range(total):
    s=sweep(s)
    if k%every==every-1:
        m=s.reshape(R,-1).mean(axis=1)
        M.append(m.astype(np.float32))
M=np.array(M)  # (nmeas, R)
np.save('/work/M16.npy',M)
print('done',time.time()-t0,M.shape)
