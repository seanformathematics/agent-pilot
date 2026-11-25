import numpy as np, time
N=16; R=2000; T=2.2
rng=np.random.default_rng(1)
s=np.where(rng.random((R,N,N))<0.5,-1,1).astype(np.int8)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
mask0=((ii+jj)%2==0); mask1=~mask0
tab=np.array([min(1.0,np.exp(-2*h/T)) for h in range(-4,5)])
def half(s,mask):
    nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
    h=(s*nn)
    p=tab[(h+4)]
    acc=(rng.random(s.shape,dtype=np.float32)<p)&mask
    s[acc]*=-1  # flip
    return s
t=time.time()
for _ in range(100):
    half(s,mask0); half(s,mask1)
dt=time.time()-t
print("per sweep ms", dt/100*1000)
