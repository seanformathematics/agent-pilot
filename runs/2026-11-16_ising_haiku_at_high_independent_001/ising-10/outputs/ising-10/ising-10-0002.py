import numpy as np, time
N=16
def make_masks():
    i,j=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    return ((i+j)%2==0), ((i+j)%2==1)
mA,mB=make_masks()
def sweep(s,T,rng,expt):
    for m in (mA,mB):
        nn = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE = 2*s*nn  # in {-8..8}
        acc = rng.random(s.shape) < expt[dE+8]
        flip = acc & m
        s[flip] *= -1
    return s
R=1000
rng=np.random.default_rng(1)
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
for T in [2.269]:
    expt=np.zeros(17)
    for d in range(-8,9): expt[d+8]=min(1.0,np.exp(-d/T))
    t=time.time()
    for k in range(100): sweep(s,T,rng,expt)
    print("100 sweeps R=1000:", time.time()-t)
