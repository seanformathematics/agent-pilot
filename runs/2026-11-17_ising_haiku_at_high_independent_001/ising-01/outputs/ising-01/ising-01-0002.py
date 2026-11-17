
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16
def run(T, R, nburn, nmeas, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    # acceptance table for dE = 2*s*nn, s*nn in {-4..4}
    acc=np.zeros(9)
    for x in range(9):
        dE=2*(x-4)
        acc[x]=1.0 if dE<=0 else np.exp(-dE/T)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    def sweep():
        for p in (0,1):
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            sn=(s*nn).astype(np.int16)+4  # 0..8
            a=acc[sn]
            u=rng.random((R,N,N))
            flip=masks[p][None]&(u<a)
            s[flip]*=-1
    for _ in range(nburn): sweep()
    S2=np.zeros(R); S4=np.zeros(R); cnt=0
    for _ in range(nmeas):
        sweep()
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2=m*m
        S2+=m2; S4+=m2*m2; cnt+=1
    return S2/cnt, S4/cnt
t=time.time()
m2,m4=run(2.269,1000,50,50,1)
print('time',time.time()-t)
