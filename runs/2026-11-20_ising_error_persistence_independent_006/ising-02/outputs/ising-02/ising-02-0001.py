
import numpy as np, time
N=16; R=1000
rng=np.random.default_rng(12345)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T,rng):
    for par in (0,1):
        nb=(np.roll(s,1,1).astype(np.int16)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-2+2,2)*0+np.roll(s,-1,2))
        # neighbours: up,down,left,right
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int16)
        dE=2*s*nb
        acc=(dE<=0)|(rng.random(s.shape)<np.exp(-np.clip(dE,0,None)/T))
        upd=acc&masks[par]
        s[upd]*=-1
    return s
t=time.time()
for _ in range(20): sweep(s,2.27,rng)
print("time per sweep",(time.time()-t)/20)
