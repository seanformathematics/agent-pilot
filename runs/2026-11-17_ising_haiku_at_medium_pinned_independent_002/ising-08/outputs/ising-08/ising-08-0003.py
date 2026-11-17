
import numpy as np, time, os
N=16; R=1000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def run(T, seed, neq=3000, nmeas=8000):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    p={4:np.exp(-4/T),8:np.exp(-8/T)}
    def sweep():
        for m in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nb
            r=rng.random(s.shape)
            flip=(dE<=0)|((dE==4)&(r<p[4]))|((dE==8)&(r<p[8]))
            flip&=m[None]
            s[flip]*=-1
    for _ in range(neq): sweep()
    M2=np.zeros(R); M4=np.zeros(R); cnt=0
    for _ in range(nmeas):
        sweep()
        m=s.sum(axis=(1,2))/N**2
        M2+=m**2; M4+=m**4; cnt+=1
    return M2/cnt, M4/cnt
t0=time.time()
for k,T in enumerate([1.9,2.0,2.1,2.2,2.3,2.4]):
    a2,a4=run(T,seed=200+k)
    np.savez(f'/work/res_{T}.npz',m2=a2,m4=a4)
    print(T, time.time()-t0, flush=True)
