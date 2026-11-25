
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16
def make_masks(N):
    i,j=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    return ((i+j)%2==0), ((i+j)%2==1)
EV,OD=make_masks(N)
def sweep(s,T,rng,EV,OD):
    # s: (R,N,N) int8
    for mask in (EV,OD):
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*nn  # energy change when flipping (J=1): dE=2 s h
        acc=(dE<=0)|(rng.random(s.shape,dtype=np.float32)<np.exp(-np.maximum(dE,0)/T).astype(np.float32))
        flip=acc&mask
        s[flip]*=-1
    return s
R=2000
rng=np.random.default_rng(12345)
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
t=time.time()
for _ in range(50): s=sweep(s,2.269,rng,EV,OD)
print('time per sweep (R=2000):',(time.time()-t)/50)
m=s.reshape(R,-1).mean(1); print(np.abs(m).mean())
