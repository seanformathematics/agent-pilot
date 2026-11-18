
import numpy as np, time, os
N=16; R=512
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def run(T, seed, neq=3000, nmeas=20000, blk=10):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep(s):
        for k in range(2):
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=(2*s*nb).astype(np.float64)
            acc=(rng.random(s.shape)<np.exp(-dE/T))&masks[k]
            s=np.where(acc,-s,s).astype(np.int8)
        return s
    for _ in range(neq): s=sweep(s)
    nb_=nmeas//blk
    m2b=np.zeros((R,nb_)); m4b=np.zeros((R,nb_))
    for b in range(nb_):
        a2=np.zeros(R); a4=np.zeros(R)
        for _ in range(blk):
            s=sweep(s)
            m=s.reshape(R,-1).mean(axis=1)
            a2+=m*m; a4+=m**4
        m2b[:,b]=a2/blk; m4b[:,b]=a4/blk
    return m2b,m4b
os.makedirs('/work',exist_ok=True)
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1]
for idx,T in enumerate(Ts):
    t=time.time()
    seed=1000+idx
    m2b,m4b=run(T,seed)
    np.savez(f'/work/res_{T}.npz',m2b=m2b,m4b=m4b,seed=seed)
    m2=m2b.mean(); m4=m4b.mean()
    print(T, "U_naive", 1-m4/(3*m2*m2), "time", time.time()-t, flush=True)
