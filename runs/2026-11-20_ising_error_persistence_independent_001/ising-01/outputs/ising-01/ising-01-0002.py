
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16; R=2000; T=2.2
rng=np.random.default_rng(1)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==c) for c in (0,1)]
beta=1/T
ptab=np.zeros(9)
for x in range(-4,5): ptab[x+4]=np.exp(-2*beta*x) if x>0 else 1.0
def sweep(s):
    for m in masks:
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        x=s*h
        p=ptab[(x+4)]
        flip=(rng.random(s.shape)<p)&m
        s[flip]*=-1
    return s
t=time.time()
for k in range(20): sweep(s)
print('per sweep',(time.time()-t)/20)
