import numpy as np, time
N=16; R=1000; T=2.269
rng=np.random.default_rng(1)
S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def sweep(S,T):
    for p in (0,1):
        nn=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2))
        dE=2*S*nn
        acc=np.exp(-dE/T)
        r=rng.random(S.shape)
        flip=(dE<=0)|(r<acc)
        flip&=masks[p]
        S=np.where(flip,-S,S)
    return S
t=time.time()
for k in range(50): S=sweep(S,T)
dt=time.time()-t
print("per sweep (R=1000):",dt/50)
