
import time, numpy as np, json, os
os.makedirs('/work',exist_ok=True)
N=16; R=512
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def run(T, seed, burn=2000, meas=10000):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    beta=1.0/T
    # precompute acceptance table for dE in {0,4,8} (dE = 2*s*nb, nb in {-4,-2,0,2,4})
    acc_tab={d:np.exp(-beta*d) for d in (4,8)}
    def sweep():
        for m in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nb
            p=np.where(dE<=0,1.0,np.exp(-beta*np.clip(dE,0,None)))
            flip=m[None]&(rng.random(s.shape)<p)
            s[flip]*=-1
    for _ in range(burn): sweep()
    M=np.empty((meas,R))
    for k in range(meas):
        sweep()
        M[k]=s.reshape(R,-1).mean(axis=1)
    return M
results={}
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
t0=time.time()
for idx,T in enumerate(Ts):
    M=run(T, seed=1000+idx)
    np.save(f'/work/M_T{T}.npy',M)
    m2=(M**2).mean(); m4=(M**4).mean()
    print(T, "U_raw=",1-m4/(3*m2**2), "time",time.time()-t0, flush=True)
    if time.time()-t0>480: break
