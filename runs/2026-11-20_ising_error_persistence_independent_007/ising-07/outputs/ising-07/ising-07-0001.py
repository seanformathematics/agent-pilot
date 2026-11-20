
import numpy as np, time
N=16
def run(T, R, nsweep, nequil, seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    acc={4:np.exp(-4/T),8:np.exp(-8/T)}
    def sweep(s):
        for p in (0,1):
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nn  # energy change for flipping s
            prob=np.where(dE<=0,1.0,np.where(dE==4,acc[4],np.where(dE==8,acc[8],0.0)))
            u=rng.random((R,N,N))
            flip=(u<prob)&masks[p]
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nequil): s=sweep(s)
    m2=np.zeros(R); m4=np.zeros(R); cnt=0
    for _ in range(nsweep):
        s=sweep(s)
        m=s.reshape(R,-1).mean(1)
        m2+=m*m; m4+=m**4; cnt+=1
    return m2/cnt, m4/cnt
t=time.time()
m2,m4=run(2.3,1000,100,0,1)
print(time.time()-t)
