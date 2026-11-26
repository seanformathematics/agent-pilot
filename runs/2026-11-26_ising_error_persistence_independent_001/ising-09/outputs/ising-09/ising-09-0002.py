
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16
def run(T, R, nburn, nmeas, every, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    # acceptance table indexed by s*nn in {-4,-2,0,2,4}
    acc={k: np.exp(-2.0*k/T) for k in [-4,-2,0,2,4]}
    tab=np.zeros(9)
    for k in [-4,-2,0,2,4]: tab[k+4]=min(1.0,acc[k])
    def sweep(s):
        for m in masks:
            nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            sn=(s*nn).astype(np.int16)+4
            p=tab[sn]
            flip=(rng.random(s.shape)<p)&m
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    t0=time.time()
    for _ in range(nburn): s=sweep(s)
    tb=time.time()-t0
    sm2=np.zeros(R); sm4=np.zeros(R); cnt=0
    ms=[]
    for k in range(nmeas):
        for _ in range(every): s=sweep(s)
        m=s.reshape(R,-1).mean(1)
        ms.append(m)
    return np.array(ms), tb, time.time()-t0
t=time.time()
ms,tb,tt=run(2.269,500,20,10,1,1)
print("time burn 20 sweeps R=500:",tb,"total",tt)
