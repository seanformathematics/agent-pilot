
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16
def make_masks():
    i,j=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    return ((i+j)%2==0), ((i+j)%2==1)
mA,mB=make_masks()
def sweep(s,T,rng,mA,mB):
    for mask in (mA,mB):
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nn
        p=np.exp(-np.clip(dE,0,None)/T)
        acc=(dE<=0)|(rng.random(s.shape)<p)
        flip=acc&mask
        s=np.where(flip,-s,s)
    return s
rng=np.random.default_rng(12345)
R=512
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
t=time.time()
for k in range(200):
    s=sweep(s,2.3,rng,mA,mB)
print("time per sweep (R=512):",(time.time()-t)/200)
