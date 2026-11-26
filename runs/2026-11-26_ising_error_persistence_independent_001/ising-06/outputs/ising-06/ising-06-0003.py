import numpy as np, time, os
N=16
Ts=np.array([1.9,2.0,2.1,2.2,2.3,2.4])
K=2000
R=len(Ts)*K
i,j=np.indices((N,N))
masks=[((i+j)%2==p) for p in (0,1)]
SEED=2024
rng=np.random.default_rng(SEED)
T=np.repeat(Ts,K)
beta=(1.0/T).astype(np.float32).reshape(-1,1,1)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
def sweep(s):
    for m in masks:
        nb = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE = 2*s*nb
        acc = np.exp(-beta*np.maximum(dE,0).astype(np.float32))
        r = rng.random(s.shape,dtype=np.float32)
        flip = (r<acc) & m
        s[flip] *= -1
t0=time.time()
log=[]
for k in range(1500):
    sweep(s)
    if (k+1)%100==0:
        mm=s.reshape(R,-1).mean(1)
        m2=(mm**2).reshape(len(Ts),K).mean(1)
        nb=(np.roll(s,1,1)*s+np.roll(s,1,2)*s).reshape(R,-1).mean(1)
        e=(-nb).reshape(len(Ts),K).mean(1)
        log.append((k+1,m2,e))
        print(k+1, np.round(m2,3), np.round(e,3), round(time.time()-t0,1), flush=True)
np.save('/work/state_s.npy', s)
np.save('/work/state_rng.npy', np.array([0]))  # placeholder
