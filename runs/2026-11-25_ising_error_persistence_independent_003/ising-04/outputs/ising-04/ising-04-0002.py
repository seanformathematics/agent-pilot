
import numpy as np, time, os
os.makedirs("/work", exist_ok=True)
N=16
def make(R, rng):
    s = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
    return s
ii,jj = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==c) for c in (0,1)]
def sweep(s, T, rng, probtab_cache={}):
    key=T
    if key not in probtab_cache:
        # index k = s*nn + 4, s*nn in {-4,-2,0,2,4}; dE = 2*s*nn
        tab=np.zeros(9)
        for sn in (-4,-2,0,2,4):
            dE=2*sn
            tab[sn+4]=1.0 if dE<=0 else np.exp(-dE/T)
        probtab_cache[key]=tab
    tab=probtab_cache[key]
    for c in (0,1):
        nn = (np.roll(s,1,axis=1)+np.roll(s,-1,axis=1)+np.roll(s,1,axis=2)+np.roll(s,-1,axis=2))
        sn = (s*nn).astype(np.int16)+4
        p = tab[sn]
        u = rng.random(s.shape)
        flip = (u < p) & masks[c]
        s = np.where(flip, -s, s).astype(np.int8)
    return s
rng=np.random.default_rng(12345)
R=1000
s=make(R,rng)
t=time.time()
for k in range(100):
    s=sweep(s,2.27,rng)
print("time per sweep R=1000:", (time.time()-t)/100)
