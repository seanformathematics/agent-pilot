
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
rng=np.random.default_rng(1)
R,N=1000,16
S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(S,T,rng):
    for par in (0,1):
        nn=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)).astype(np.int8)
        dE=2*S*nn
        p=np.exp(-dE/T)
        acc=(dE<=0)|(rng.random(S.shape)<p)
        S[:,masks[par]] = np.where(acc,-S,S)[:,masks[par]]
    return S
t=time.time()
for _ in range(200): S=sweep(S,2.2,rng)
print('time per sweep (R=1000):',(time.time()-t)/200)
print(np.abs(S.mean(axis=(1,2))).mean())
