
import numpy as np, time
N=16
def make_mask():
    i,j=np.indices((N,N)); return ((i+j)%2==0)
MASK=make_mask()
def sweep(s,T,rng,mask):
    for m in (mask,~mask):
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nn  # energy change for flipping s (J=1)
        p=np.exp(-np.maximum(dE,0)/T)
        acc=(dE<=0)|(rng.random(s.shape)<p)
        s[acc & m[None]] *= -1
    return s
rng=np.random.default_rng(1)
R=1000
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
t=time.time()
for k in range(100): sweep(s,2.3,rng,MASK)
print("time per sweep (R=1000):",(time.time()-t)/100)
