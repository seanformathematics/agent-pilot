import numpy as np, time
N=16; R=512; T=2.3; beta=1/T
rng=np.random.default_rng(12345)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
acc={d:np.exp(-beta*d) for d in (4,8)}
def sweep(s):
    for m in masks:
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*nn  # in {-8,-4,0,4,8}
        u=rng.random(s.shape)
        flip=(dE<=0)|((dE==4)&(u<acc[4]))|((dE==8)&(u<acc[8]))
        flip&=m
        s[flip]*=-1
    return s
t=time.time()
for _ in range(100): sweep(s)
print("time per sweep (R=512):",(time.time()-t)/100)