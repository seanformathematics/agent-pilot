import time, json, numpy as np
R,L=1000,16
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
ii,jj=np.meshgrid(np.arange(L),np.arange(L),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
out={}
def run(T,seed,burn=3000,meas=5000):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,L,L))
    beta=1/T
    # precompute acceptance probs for dE=4,8 (dE=2*s*nb in {-8..8})
    p={d:np.exp(-beta*d) for d in (4,8)}
    def sweep():
        for m in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nb
            r=rng.random(s.shape,dtype=np.float32)
            acc=(dE<=0)|((dE==4)&(r<p[4]))|((dE==8)&(r<p[8]))
            acc&=m
            s[acc]*=-1
    for _ in range(burn): sweep()
    m2=np.zeros(R); m4=np.zeros(R); 
    for _ in range(meas):
        sweep()
        mm=s.reshape(R,-1).mean(1)
        m2+=mm**2; m4+=mm**4
    return m2/meas, m4/meas
t0=time.time()
res={}
for T in Ts[:4]:
    m2,m4=run(T,seed=int(T*1000))
    res[T]=(m2.tolist(),m4.tolist())
    print(T, time.time()-t0, flush=True)
json.dump({str(k):v for k,v in res.items()},open('/work/part1.json','w'))
