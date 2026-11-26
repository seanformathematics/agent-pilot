import numpy as np, time
N=16; R=512
rng=np.random.default_rng(1)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==c) for c in (0,1)]
T=2.269
def sweep(s,T):
    for c in (0,1):
        nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE=2*s*nn
        p=np.exp(-dE/T)
        acc=(rng.random(s.shape)<p)&masks[c]
        s=np.where(acc,-s,s).astype(np.int8)
    return s
t=time.time()
for _ in range(200): s=sweep(s,T)
dt=time.time()-t
print("per sweep ms",dt/200*1e3, "sample-sweeps/s", R*200/dt)
