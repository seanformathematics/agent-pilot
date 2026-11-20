import numpy as np, time
N=16
def run(T, R, nburn, nmeas, seed):
    rng=np.random.default_rng(seed)
    S=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    # acceptance for dE = 2*s*nn, s*nn in {-4,-2,0,2,4}; index by s*nn+4
    acc=np.zeros(9)
    for x in range(-4,5):
        dE=2*x
        acc[x+4]=1.0 if dE<=0 else np.exp(-dE/T)
    def sweep():
        for p in (0,1):
            nn=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)).astype(np.int8)
            snn=(S*nn).astype(np.int8)+4
            a=acc[snn]
            flip=masks[p][None]&(rng.random((R,N,N))<a)
            S[flip]*=-1
    for _ in range(nburn): sweep()
    m2s=np.zeros(R); m4s=np.zeros(R)
    for _ in range(nmeas):
        sweep()
        m=S.reshape(R,-1).mean(1)
        m2s+=m*m; m4s+=m**4
    return m2s/nmeas, m4s/nmeas
t=time.time()
m2,m4=run(2.3,500,500,1000,12345)
print("time",time.time()-t)
U_rep=1-m4/(3*m2**2)
print("U mean",1-m4.mean()/(3*m2.mean()**2))
print("per-replica U std",U_rep.std(), "naive SE", U_rep.std()/np.sqrt(len(U_rep)))