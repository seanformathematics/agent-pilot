
import time, importlib, numpy as np
print(np.__version__)
for m in ["numba"]:
    try:
        importlib.import_module(m); print(m, "available")
    except Exception as e: print(m, "missing", e)

N=16; R=512
rng=np.random.default_rng(1)
s=rng.choice([-1,1],size=(R,N,N)).astype(np.int8)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,beta,rng):
    for m in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nb
        acc=(rng.random(s.shape)<np.exp(-beta*np.clip(dE,0,None)))
        flip=m[None]&((dE<=0)|acc)
        s[flip]*=-1
    return s
t=time.time()
for k in range(200):
    sweep(s,1/2.2,rng)
print("time per sweep (R=512):",(time.time()-t)/200)
