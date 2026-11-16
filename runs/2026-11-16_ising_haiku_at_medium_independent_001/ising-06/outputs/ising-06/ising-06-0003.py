
import numpy as np, time, json
N=16
def run(T, R, nequil, nmeas, every, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    def sweep():
        for p in (0,1):
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int16)
            dE=2*s.astype(np.int16)*nn
            prob=np.where(dE<=0,1.0,np.exp(-np.maximum(dE,0)/T))
            flip=(rng.random(s.shape)<prob)&masks[p]
            s[flip]*=-1
    for _ in range(nequil): sweep()
    M=[]
    for k in range(nmeas):
        for _ in range(every): sweep()
        M.append(s.reshape(R,-1).mean(axis=1))
    return np.array(M)
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1]
out={}
for i,T in enumerate(Ts):
    t0=time.time()
    M=run(T,1024,1000,800,5,100+i)
    np.save(f'/work/M_{T}.npy',M)
    print(T, time.time()-t0, flush=True)
