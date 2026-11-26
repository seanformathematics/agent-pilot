
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
R=1000
rng=np.random.default_rng(12345)
Tarr=np.repeat(np.array(Ts),R)
beta=(1.0/Tarr).astype(np.float32).reshape(-1,1,1)
S=rng.choice(np.array([-1,1],dtype=np.int8),size=(len(Tarr),N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(S):
    for m in masks:
        nn=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)).astype(np.float32)
        dE=2*S*nn
        acc=(dE<=0)|(rng.random(S.shape,dtype=np.float32)<np.exp(-beta*dE))
        upd=acc&m
        S[upd]*=-1
    return S
t=time.time()
for _ in range(20): sweep(S)
print('time per sweep',(time.time()-t)/20, 'shape',S.shape)
