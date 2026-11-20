
import numpy as np, time
N=16
def run(T, R, nburn, nmeas, seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,-1.0,1.0).astype(np.float32)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    # acceptance table: dE = 2*s*nn, nn in {-4,-2,0,2,4}
    acc={}
    for sn in range(-4,5):
        for d in [-8,-4,0,4,8]:
            pass
    def sweep(s):
        for p in (0,1):
            nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nn
            prob=np.exp(-np.maximum(dE,0)/T)
            r=rng.random(s.shape,dtype=np.float32)
            flip=(r<prob)&masks[p]
            s=np.where(flip,-s,s)
        return s
    for _ in range(nburn): s=sweep(s)
    m2=np.zeros((nmeas,R)); m4=np.zeros((nmeas,R))
    for k in range(nmeas):
        s=sweep(s)
        m=s.mean(axis=(1,2))
        m2[k]=m*m; m4[k]=m**4
    return m2,m4
t=time.time()
m2,m4=run(2.269,1000,20,20,1)
print("time per sweep (R=1000):",(time.time()-t)/40)
