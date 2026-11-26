
import numpy as np, time
N=16
def simulate(T, R, nburn, nmeas, every, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    tab=np.zeros(9)
    for k in [-4,-2,0,2,4]: tab[k+4]=min(1.0,np.exp(-2.0*k/T))
    def sweep(s):
        for m in masks:
            nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            p=tab[(s*nn).astype(np.int16)+4]
            flip=(rng.random(s.shape)<p)&m
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nburn): s=sweep(s)
    out=np.zeros((nmeas,R),dtype=np.float32)
    for k in range(nmeas):
        for _ in range(every): s=sweep(s)
        out[k]=s.reshape(R,-1).mean(1)
    return out
for T,seed in [(2.3,1005),(2.4,1006)]:
    t=time.time()
    ms=simulate(T,1000,3000,4000,5,seed)
    np.save(f'/work/ms_T{T}_N{N}.npy',ms)
    m2=ms.astype(float)**2
    blocks=m2.reshape(8,500,-1).mean(axis=(1,2))
    U=1-np.mean(ms.astype(float)**4)/(3*np.mean(m2)**2)
    print(T,"time",round(time.time()-t),"block<m2>",np.round(blocks,4),"U",U,flush=True)
