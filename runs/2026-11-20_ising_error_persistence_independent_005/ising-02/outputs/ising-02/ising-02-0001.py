
import numpy as np, time
N=16
def run(T, R, nsweep, nburn, seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==c) for c in (0,1)]
    acc={dE:np.exp(-dE/T) for dE in (4,8)}
    def sweep(s):
        for m in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            dE=2*s*nb  # energy change for flipping (E=-sum ss)
            r=rng.random(s.shape)
            flip=m & ((dE<=0) | ((dE==4)&(r<acc[4])) | ((dE==8)&(r<acc[8])))
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nburn): s=sweep(s)
    M2=np.zeros(R); M4=np.zeros(R); cnt=0
    for _ in range(nsweep):
        s=sweep(s)
        M=s.reshape(R,-1).sum(1).astype(float)
        M2+=M**2; M4+=M**4; cnt+=1
    return M2/cnt, M4/cnt
t=time.time()
m2,m4=run(2.269,1000,20,0,1)
print(time.time()-t)
