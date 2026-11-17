import numpy as np, time
N=16
R=500
T=2.269
rng=np.random.default_rng(12345)
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T,rng):
    for par in (0,1):
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*nn  # in {-8,-4,0,4,8}
        p=np.exp(-dE.astype(np.float64)/T)
        acc=(rng.random(s.shape)<p)&masks[par]
        s=np.where(acc,-s,s).astype(np.int8)
    return s
t0=time.time()
for k in range(100):
    s=sweep(s,T,rng)
print("time per sweep", (time.time()-t0)/100)
