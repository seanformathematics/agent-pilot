import numpy as np,time
N=16;R=1000;T=2.27
rng=np.random.default_rng(1)
S=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def half(S,mask):
    nb=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2))
    dE=2*S*nb
    prob=np.exp(-dE/T)
    u=rng.random(S.shape,dtype=np.float32)
    flip=(u<prob)&mask
    S[flip]*=-1
t0=time.time()
for _ in range(100):
    half(S,masks[0]);half(S,masks[1])
print((time.time()-t0)/100)