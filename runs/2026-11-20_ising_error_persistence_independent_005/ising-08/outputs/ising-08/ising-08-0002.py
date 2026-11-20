import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16; R=400
def run(T, seed, nequil=1500, nmeas=3000):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    # acceptance table indexed by s*nb+4 where nb=sum of 4 neighbours, dE=2*s*nb
    k=np.arange(-4,5)
    acc=np.minimum(1.0,np.exp(-2*k*1.0/T))
    accT=acc.astype(np.float64)
    def sweep(s):
        for m in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            idx=(s*nb).astype(np.int16)+4
            p=accT[idx]
            u=rng.random(s.shape)
            flip=(u<p)&m
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nequil): s=sweep(s)
    M2=np.zeros((nmeas,R)); M4=np.zeros((nmeas,R)); AM=np.zeros((nmeas,R))
    for t in range(nmeas):
        s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1)
        M2[t]=m*m; M4[t]=m**4; AM[t]=np.abs(m)
    return M2,M4,AM
t0=time.time()
M2,M4,AM=run(2.269,1,nequil=100,nmeas=100)
print('time for 200 sweeps',time.time()-t0)
