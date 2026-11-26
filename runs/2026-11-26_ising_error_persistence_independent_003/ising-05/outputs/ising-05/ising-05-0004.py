
import numpy as np, time
N=16; R=512
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def run(T, seed, NEQ=3000, NMEAS=20000, BLK=200):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep(s):
        for m in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            dE=2*s*nb
            acc=(dE<=0)|(rng.random(s.shape)<np.exp(-np.maximum(dE,0)/T))
            s[acc&m]*=-1
    for k in range(NEQ): sweep(s)
    nbk=NMEAS//BLK
    m2b=np.zeros((nbk,R)); m4b=np.zeros((nbk,R))
    for k in range(NMEAS):
        sweep(s)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        b=k//BLK
        m2b[b]+=m*m; m4b[b]+=m**4
    m2b/=BLK; m4b/=BLK
    np.savez(f"/work/T{T:.2f}.npz",m2b=m2b,m4b=m4b)
t=time.time()
for T,seed in [(2.2,1220),(2.3,1230),(2.4,1240)]:
    run(T,seed); print("done",T,time.time()-t, flush=True)
