
import numpy as np, time, os
os.makedirs('/work', exist_ok=True)
N=16
def run(T, R, nequil, nmeas, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    # acceptance table for dE = 4,8 (possible positive dE values for 2D Ising: 4 and 8 times s*nn...)
    def sweep():
        for p in (0,1):
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nn
            acc=(dE<=0)|(rng.random((R,N,N))<np.exp(-np.maximum(dE,0)/T))
            s[:]=np.where(masks[p]&acc,-s,s)
    for _ in range(nequil): sweep()
    m1=np.zeros(R); m2=np.zeros(R); m4=np.zeros(R)
    for _ in range(nmeas):
        sweep()
        m=s.sum(axis=(1,2))/(N*N)
        m2+=m*m; m4+=m**4
    return m2/nmeas, m4/nmeas
t=time.time()
a,b=run(2.0,1000,50,50,1)
print("time for 100 sweeps R=1000:",time.time()-t)
