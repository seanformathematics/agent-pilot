
import numpy as np, time, json
N=16; R=2000
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
burn=1000; meas=4000; nblk=20
res={}
t0=time.time()
for k,T in enumerate(Ts):
    seed=1000+k
    rng=np.random.default_rng(seed)
    S=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    def sweep(S):
        for m in masks:
            nb=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2))
            dE=2*S*nb
            acc=(dE<=0)|(rng.random(S.shape,dtype=np.float32)<np.exp(-np.maximum(dE,0)/T).astype(np.float32))
            S=np.where(m[None]&acc,-S,S).astype(np.int8)
        return S
    for _ in range(burn):
        S=sweep(S)
    # per-replica block sums of m^2, m^4, m (block-level)
    s2=np.zeros((R,nblk)); s4=np.zeros((R,nblk)); s1=np.zeros((R,nblk))
    per=meas//nblk
    for b in range(nblk):
        for _ in range(per):
            S=sweep(S)
            m=S.reshape(R,-1).mean(axis=1)
            s2[:,b]+=m**2; s4[:,b]+=m**4; s1[:,b]+=np.abs(m)
    s2/=per; s4/=per; s1/=per
    res[T]={'s2':s2.tolist(),'s4':s4.tolist(),'s1':s1.tolist(),'seed':seed}
    print(T, time.time()-t0, flush=True)
json.dump(res,open('/work/res.json','w')) if __import__('os').path.isdir('/work') else json.dump(res,open('/tmp/res.json','w'))
