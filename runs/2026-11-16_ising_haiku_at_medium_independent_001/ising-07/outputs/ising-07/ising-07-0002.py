import numpy as np, time, json, os
from multiprocessing import Pool
N=16; R=800; EQ=3000; MEAS=6000
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def run(args):
    T,seed=args
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    acc_table=None
    def sweep():
        for mk in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.float32)
            dE=2*s*nb
            p=np.exp(-np.maximum(dE,0)/T)
            flip=mk[None]&(rng.random(size=s.shape,dtype=np.float32)<p)
            s[flip]*=-1
    for _ in range(EQ): sweep()
    m2=np.zeros(R); m4=np.zeros(R); m1=np.zeros(R)
    for _ in range(MEAS):
        sweep()
        m=s.sum(axis=(1,2),dtype=np.float64)/(N*N)
        m2+=m*m; m4+=m**4; m1+=np.abs(m)
    return T,(m2/MEAS).tolist(),(m4/MEAS).tolist(),(m1/MEAS).tolist()
if __name__=='__main__':
    t0=time.time()
    with Pool(8) as pool:
        res=pool.map(run,[(T,1000+k) for k,T in enumerate(Ts)])
    json.dump(res,open('/work/res1.json','w'))
    print('done',time.time()-t0)
