
import numpy as np, time
N=16
def make(R, rng):
    s = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
    return s
ii,jj = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
mask0 = ((ii+jj)%2==0)
mask1 = ~mask0
def sweep(s, T, rng, beta_tab):
    R = s.shape[0]
    for mask in (mask0, mask1):
        nb = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        # dE = 2*s*nb ; accept if dE<=0 or u < exp(-dE/T)
        dE = 2*s*nb
        u = rng.random(s.shape, dtype=np.float32)
        acc = (dE <= 0) | (u < beta_tab[np.clip(dE,0,8)//2 if False else np.clip(dE+8,0,16)])
        flip = acc & mask
        s[flip] *= -1
    return s
rng = np.random.default_rng(1)
for R in [256,1024]:
    s = make(R,rng)
    T=2.269
    beta_tab = np.exp(-np.arange(-8,9)/T).astype(np.float32)  # index dE+8
    # simpler: table over dE in [-8..8]; use index dE+8
    t=time.time()
    for k in range(50):
        sweep(s,T,rng,beta_tab)
    dt=time.time()-t
    print(R, dt/50, "s per sweep;", R*50/dt, "replica-sweeps/s")
