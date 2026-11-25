
import numpy as np, time
N=16; R=512
rng=np.random.default_rng(12345)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
mA=((ii+jj)%2==0)[None]; mB=~mA
def sweep(s,p):
    for mask in (mA,mB):
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=(2*s*nn).astype(np.int8)
        acc=rng.random(s.shape)< p[dE+8]
        flip=mask&acc
        s=np.where(flip,-s,s).astype(np.int8)
    return s
T=2.269
p=np.array([min(1.0,np.exp(-d/T)) for d in range(-8,9)])
t=time.time()
for _ in range(200): s=sweep(s,p)
print("time per sweep (R=512):",(time.time()-t)/200)
