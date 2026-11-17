
import numpy as np, time, json, os
os.makedirs('/work',exist_ok=True)
N=16; R=400; NEQ=2000; NMEAS=10000
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
results={}
t0=time.time()
for k,T in enumerate(Ts):
    seed=1000+k
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep(s):
        for m in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            dE=2*s*nb
            acc=(dE<=0)|(rng.random(s.shape)<np.exp(-np.clip(dE,0,None)/T))
            s[acc&m]*=-1
    for _ in range(NEQ): sweep(s)
    m2=np.zeros(R); m4=np.zeros(R)
    for _ in range(NMEAS):
        sweep(s)
        mm=s.reshape(R,-1).mean(axis=1)
        m2+=mm**2; m4+=mm**4
    m2/=NMEAS; m4/=NMEAS
    # per-replica U-like quantities; jackknife over replicas
    def Uf(a2,a4): return 1-np.mean(a4)/(3*np.mean(a2)**2)
    U=Uf(m2,m4)
    jk=np.empty(R)
    idx=np.arange(R)
    for r in range(R):
        keep=idx!=r
        jk[r]=Uf(m2[keep],m4[keep])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    results[T]=dict(seed=seed,U=float(U),U_err=float(err),n_samples=R*NMEAS)
    print(T,U,err,time.time()-t0,flush=True)
json.dump(results,open('/work/results.json','w'),indent=1)
